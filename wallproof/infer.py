"""Stage 1: run the pretrained models once per room and cache their outputs.

Models (all pretrained, no fine-tuning):
  - Mask2Former Swin-L, ADE20K semantic  -> per-pixel class + soft "wall" probability
  - Apple Depth Pro                      -> metric depth (m) + estimated focal length (px)
  - Depth Anything V2 Metric Indoor L    -> second metric depth opinion (scale cross-check)
  - MoGe-2 ViT-L (MIT licence)          -> metric depth + focal length, commercially usable
                                           candidate to replace Depth Pro
  - Intrinsic image decomposition v2.1   -> lighting (shading) separated from surface colour
    (Careaga & Aksoy, compphoto/Intrinsic) so dark paint or brick is not mistaken for shadow

Outputs one .npz per room in cache/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    DepthProForDepthEstimation,
    DepthProImageProcessorFast,
    Mask2FormerForUniversalSegmentation,
)

SEG_ID = "facebook/mask2former-swin-large-ade-semantic"
DEPTHPRO_ID = "apple/DepthPro-hf"
DA2_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

torch.set_grad_enabled(False)
torch.set_num_threads(max(1, torch.get_num_threads()))


def segment(img: Image.Image, proc, model):
    W, H = img.size
    inputs = proc(images=img, return_tensors="pt")
    out = model(**inputs)
    cls = out.class_queries_logits.softmax(-1)[..., :-1]  # (1, Q, C)
    masks = out.masks_queries_logits.sigmoid()  # (1, Q, h, w)
    # Upsample mask logits to full resolution before combining -> sharper edges.
    masks = torch.nn.functional.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)
    seg = torch.einsum("bqc,bqhw->bchw", cls, masks)[0]  # (C, H, W)
    prob = seg / seg.sum(0, keepdim=True).clamp(min=1e-6)
    labels = prob.argmax(0).numpy().astype(np.uint8)
    wall_prob = prob[0].numpy().astype(np.float16)  # ADE20K class 0 == wall
    return labels, wall_prob


def depth_pro(img: Image.Image, proc, model):
    W, H = img.size
    inputs = proc(images=img, return_tensors="pt")
    out = model(**inputs)
    r = proc.post_process_depth_estimation(out, target_sizes=[(H, W)])[0]
    return r["predicted_depth"].numpy().astype(np.float32), float(r["focal_length"])


def depth_anything(img: Image.Image, proc, model):
    W, H = img.size
    inputs = proc(images=img, return_tensors="pt")
    out = model(**inputs)
    r = proc.post_process_depth_estimation(out, target_sizes=[(H, W)])[0]
    return r["predicted_depth"].numpy().astype(np.float32)


def intrinsic_shading(img: Image.Image, models):
    """Colour shading layer (linear RGB): image = albedo * shading."""
    from intrinsic.pipeline import run_pipeline
    arr = np.asarray(img).astype(np.float32) / 255
    r = run_pipeline(models, arr, stage=3, device="cpu")
    shd = r["hr_shd"]
    if shd.shape[:2] != arr.shape[:2]:
        import cv2
        shd = cv2.resize(shd, (arr.shape[1], arr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return shd.astype(np.float16), r["hr_alb"].astype(np.float16)


def _local_wsl_backbone():
    """The intrinsic model builds its backbones (ResNeXt-101 WSL, EfficientNet-Lite3) via
    torch.hub, which fetches GitHub repos only to construct the architectures and load ImageNet
    weights that the intrinsic checkpoints then overwrite. Build the identical torchvision / timm
    architectures locally instead (strict state-dict loading verifies the match)."""
    import torchvision.models.resnet as tvr

    real_load = torch.hub.load

    def load(repo, model, *a, **kw):
        if repo == "facebookresearch/WSL-Images":
            gw = int(model.split("_32x")[1].split("d")[0])
            return tvr.ResNet(tvr.Bottleneck, [3, 4, 23, 3], groups=32, width_per_group=gw)
        if repo == "rwightman/gen-efficientnet-pytorch":  # predecessor of timm, same layer names
            import timm
            m = timm.create_model(model, pretrained=False, exportable=kw.get("exportable", False))
            m.act1 = torch.nn.Identity()  # timm folds the stem activation into bn1 (BatchNormAct2d)
            return m
        return real_load(repo, model, *a, **kw)

    torch.hub.load = load


def add_intrinsic(paths: list[Path], cache: Path):
    """Add the shading layer to rooms cached before this model was part of the pipeline."""
    from intrinsic.pipeline import load_models
    _local_wsl_backbone()
    todo = [p for p in paths if "shading" not in np.load(cache / f"{p.stem}.npz").files]
    if not todo:
        return
    models = load_models("v2.1", stage=3, device="cpu")
    for p in todo:
        t = time.time()
        shd, alb = intrinsic_shading(Image.open(p).convert("RGB"), models)
        d = dict(np.load(cache / f"{p.stem}.npz"))
        d["shading"], d["albedo"] = shd, alb
        np.savez_compressed(cache / f"{p.stem}.npz", **d)
        print(p.name, f"intrinsic {time.time() - t:.1f}s", flush=True)


MOGE_ID = "Ruicheng/moge-2-vitl-normal"


def add_moge(paths: list[Path], cache: Path):
    """Add MoGe-2 metric depth + focal length to the cached rooms."""
    todo = [p for p in paths if "depth_moge" not in np.load(cache / f"{p.stem}.npz").files]
    if not todo:
        return
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained(MOGE_ID).eval()
    for p in todo:
        t = time.time()
        img = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255
        x = torch.from_numpy(img).permute(2, 0, 1)
        out = model.infer(x, apply_mask=False, use_fp16=False)
        depth = out["depth"].numpy().astype(np.float32)
        focal = float(out["intrinsics"][0, 0]) * img.shape[1]  # fx is normalised by image width
        d = dict(np.load(cache / f"{p.stem}.npz"))
        d["depth_moge"], d["focal_moge"] = depth, focal
        np.savez_compressed(cache / f"{p.stem}.npz", **d)
        print(p.name, f"MoGe-2 {time.time() - t:.1f}s focal={focal:.0f}px", flush=True)


def main(paths: list[Path], cache: Path):
    cache.mkdir(exist_ok=True)
    todo = [p for p in paths if not (cache / f"{p.stem}.npz").exists()]
    if not todo:
        print("all cached")
        return
    t = time.time()
    seg_p = AutoImageProcessor.from_pretrained(SEG_ID)
    seg_m = Mask2FormerForUniversalSegmentation.from_pretrained(SEG_ID).eval()
    dp_p = DepthProImageProcessorFast.from_pretrained(DEPTHPRO_ID)
    dp_m = DepthProForDepthEstimation.from_pretrained(DEPTHPRO_ID).eval()
    da_p = AutoImageProcessor.from_pretrained(DA2_ID)
    da_m = AutoModelForDepthEstimation.from_pretrained(DA2_ID).eval()
    print(f"models loaded in {time.time() - t:.0f}s", flush=True)
    for p in todo:
        img = Image.open(p).convert("RGB")
        timings = {}
        t = time.time(); labels, wall_prob = segment(img, seg_p, seg_m); timings["seg"] = time.time() - t
        t = time.time(); d_pro, focal = depth_pro(img, dp_p, dp_m); timings["depthpro"] = time.time() - t
        t = time.time(); d_da = depth_anything(img, da_p, da_m); timings["da2"] = time.time() - t
        np.savez_compressed(cache / f"{p.stem}.npz", labels=labels, wall_prob=wall_prob,
                            depth_pro=d_pro, depth_da2=d_da, focal=focal)
        print(p.name, {k: f"{v:.1f}s" for k, v in timings.items()}, f"focal={focal:.0f}px", flush=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    args = [Path(a) for a in sys.argv[1:]] or sorted((root / "data/rooms").glob("*.jpg"))
    main(args, root / "cache")
    add_intrinsic(args, root / "cache")
    add_moge(args, root / "cache")
