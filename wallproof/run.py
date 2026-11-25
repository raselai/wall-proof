"""Render before/after images + debug views + measurements for every cached room.

usage: python -m wallproof.run [room_stem ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import geometry as geo
from .products import all_products
from .render import composite

ROOT = Path(__file__).resolve().parent.parent
USE_INTRINSIC = True  # albedo from the pretrained intrinsic model corrects dark materials; False = heuristic only
STD_DOOR_M = 2.03  # standard US interior door (80 in); the test rooms are North American apartments
PLANE_COLORS = np.array([(231, 76, 60), (52, 152, 219), (46, 204, 113), (241, 196, 15), (155, 89, 182)], np.uint8)


def label_bar(img, text, h=44):
    bar = np.full((h, img.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (14, int(h * 0.68)), cv2.FONT_HERSHEY_DUPLEX, 0.85, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def depth_vis(d):
    v = np.clip((d - np.percentile(d, 1)) / (np.percentile(d, 99) - np.percentile(d, 1) + 1e-6), 0, 1)
    return cv2.cvtColor(cv2.applyColorMap((255 - v * 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def planes_vis(img, g, labels):
    out = (img * 0.45).astype(np.uint8)
    for k in range(len(g.planes)):
        m = g.plane_id == k
        out[m] = (img[m] * 0.35 + PLANE_COLORS[k % 5] * 0.65).astype(np.uint8)
    for cls, col in ((geo.FLOOR, (120, 120, 120)), (geo.CEILING, (200, 200, 200))):
        m = labels == cls
        out[m] = (img[m] * 0.6 + np.array(col) * 0.4).astype(np.uint8)
    door = np.isin(labels, geo.DOOR_IDS)
    cnt, _ = cv2.findContours(door.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnt, -1, (255, 255, 255), 3)
    return out


def door_ok(m):
    return "door" in m and 0.75 * STD_DOOR_M < m["door"]["top_m"] < 1.25 * STD_DOOR_M


def fmt_measure(m):
    parts = []
    if door_ok(m):
        parts.append(f"door {m['door']['top_m']:.2f} m")
    elif "door" in m:
        parts.append("door candidate rejected")
    if "ceiling_height_m" in m:
        parts.append(f"ceiling {m['ceiling_height_m']:.2f} m")
    return ", ".join(parts) or "no door/ceiling visible"


def process(stem: str, products, out_dir: Path):
    t0 = time.time()
    img = cv2.cvtColor(cv2.imread(str(ROOT / "data/rooms" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
    c = np.load(ROOT / "cache" / f"{stem}.npz")
    labels, wall_prob, d_pro, d_da, f = c["labels"], c["wall_prob"], c["depth_pro"], c["depth_da2"], float(c["focal"])
    alb = c["albedo"] if (USE_INTRINSIC and "albedo" in c.files) else None
    g = geo.build(labels, wall_prob, d_pro, f)
    report = {"room": stem, "image_px": [g.W, g.H], "focal_px": round(f, 1),
              "hfov_deg": round(float(np.degrees(2 * np.arctan(g.W / 2 / f))), 1), "notes": g.notes,
              "depth_pro": geo.measure(g, labels, d_pro)}
    # Second opinion: same pipeline geometry on Depth Anything V2 depth (Depth Pro focal).
    g_da = geo.build(labels, wall_prob, d_da, f)
    report["depth_anything_v2"] = geo.measure(g_da, labels, d_da)

    # Scale correction from a known reference (a door), as the product would ask the user to confirm.
    door = report["depth_pro"].get("door")
    calib = None
    if door_ok(report["depth_pro"]):
        calib = STD_DOOR_M / door["top_m"]
        report["door_calibration_factor"] = round(calib, 3)

    room_dir = out_dir / stem
    room_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(room_dir / "before.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    renders = {}
    for p in products:
        res = composite(img, g, wall_prob, p, labels=labels, albedo_lin=alb)
        renders[p.key] = res
        cv2.imwrite(str(room_dir / f"after_{p.key}.jpg"), cv2.cvtColor(res, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    if calib is not None:
        grid = next(p for p in products if p.key == "grid")
        res = composite(img, g, wall_prob, grid, scale=calib, labels=labels, albedo_lin=alb)
        renders["grid_calibrated"] = res
        cv2.imwrite(str(room_dir / "after_grid_door_calibrated.jpg"), cv2.cvtColor(res, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    # side-by-side before / after for the main product
    ba = np.hstack([label_bar(img, "BEFORE  (customer photo)"),
                    label_bar(renders["wallpaper"], f"AFTER  {products[0].name}, {products[0].note}")])
    cv2.imwrite(str(out_dir / f"{stem}__before_after.jpg"), cv2.cvtColor(ba, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])

    # debug sheet: what the models saw
    small = lambda x: cv2.resize(x, (g.W // 2, g.H // 2), interpolation=cv2.INTER_AREA)
    dp, da = report["depth_pro"], report["depth_anything_v2"]
    top = np.hstack([label_bar(small(planes_vis(img, g, labels)), f"Wall planes: {len(g.planes)} (Mask2Former + RANSAC)", 34),
                     label_bar(small(depth_vis(d_pro)), f"Depth Pro metric depth, hfov {report['hfov_deg']} deg", 34)])
    bot = np.hstack([label_bar(small(renders["grid"]), f"Scale grid, Depth Pro: {fmt_measure(dp)}", 34),
                     label_bar(small(renders.get("grid_calibrated", renders["tiles"])),
                               "Scale grid after door calibration" if calib else "Subway tile 30x10 cm", 34)])
    cv2.imwrite(str(out_dir / f"{stem}__debug.jpg"), cv2.cvtColor(np.vstack([top, bot]), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    report["depth_anything_v2_summary"] = fmt_measure(da)
    report["seconds_cpu_render"] = round(time.time() - t0, 1)
    (room_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(f"{stem}: {len(g.planes)} planes | Depth Pro: {fmt_measure(dp)} | DA-V2: {fmt_measure(da)}"
          f" | calib {calib and round(calib, 3)} | {report['seconds_cpu_render']}s")
    return report


def main(stems):
    out_dir = ROOT / "out"
    out_dir.mkdir(exist_ok=True)
    products = all_products()
    stems = stems or sorted(p.stem for p in (ROOT / "cache").glob("*.npz"))
    reports = [process(s, products, out_dir) for s in stems]
    (out_dir / "summary.json").write_text(json.dumps(reports, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
