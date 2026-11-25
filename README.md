# Wall product visualiser — proof of concept

**Live before/after gallery: https://raselai.github.io/wall-proof/**

Puts a real wall product (wallpaper, tiles) into a customer's room photo **automatically and at
true physical size**, using only pretrained models plus geometry. No generative model is involved,
so the product pattern is reproduced exactly; only its placement, perspective and lighting change.

Tested on 6 public room photos (Wikimedia Commons, see `data/rooms/CREDITS.md`) with 3 products.
Everything ran on a 4-core CPU with no GPU.

## Results gallery

`docs/` is a static before/after gallery: a slider per room, a product switcher, measurements and debug views.
Open `docs/index.html` locally, or publish it with GitHub Pages
(Settings → Pages → Deploy from branch → `main` / `docs`).

## Pipeline

```
room photo ─┬─> Mask2Former (Swin-L, ADE20K) ──> wall / floor / ceiling / door / window / furniture masks
            ├─> Apple Depth Pro ────────────────> metric depth (m) + focal length (px)
            ├─> Depth Anything V2 Metric Indoor ─> second depth opinion (scale cross-check only)
            └─> Intrinsic decomposition v2.1 ───> albedo layer (surface colour vs lighting)
                        │
                        v
   floor plane (RANSAC)  ──> "up" direction + floor height
   wall planes (2-point RANSAC, forced vertical, merged when one wall was split)
   per-pixel plane assignment
                        │
                        v
   for every wall pixel: camera ray ∩ wall plane -> (metres along wall, metres above floor)
   -> look up the product texture at that physical position (mip-mapped, no shimmer)
   -> × room shading (light falloff, sun patches, shadows) × light colour,
        corrected one-way by the intrinsic albedo so dark brick/paint is not read as shadow
   -> keep skirting boards (detected per wall as a band of constant height)
   -> composite with soft segmentation edges; furniture, doors and windows stay in front
                        │
                        v
             before/after image + measurements report
```

| Stage | Model / method | CPU time per photo |
|---|---|---|
| Segmentation | `facebook/mask2former-swin-large-ade-semantic` | ~3.5 s |
| Metric depth + focal | `apple/DepthPro-hf` | ~12 s |
| Depth cross-check | `depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf` | ~4.5 s |
| Albedo vs lighting | compphoto/Intrinsic v2.1 (Careaga & Aksoy) | ~20 s |
| Geometry + render (3 products + debug views) | numpy / OpenCV | ~12 s (unoptimised) |

On a GPU the model stages should drop to well under a second each.

## Scale check (the biggest risk)

We don't have measured ground truth for public photos. A standard US interior door (2.03 m) is the
best reference available, so it serves as the yardstick.

| Room | Reference | Depth Pro | Depth Anything V2 |
|---|---|---|---|
| room3_corner_door | door, std 2.03 m | **1.94 m (−4.4%)** | 2.25 m (+10.8%) |
| room5_door_brick | door, std 2.03 m | **2.09 m (+3.0%)** | 2.34 m (+15.3%) |
| room1_plain_wall | ceiling height | 2.10 m | 2.22 m |
| room3_corner_door | ceiling height (US typical 2.44 m) | 2.40 m | 2.77 m |
| room2_fireplace | back wall width | 4.74 m | 3.89 m |
| room2_fireplace | left wall, seen edge-on | 1.41 m | 24.0 m (plane fit failed) |

**MoGe-2 follow-up (MIT licence).** The same geometry run on MoGe-2 depth and focal length:
door 2.02 m (−0.7%) and 2.13 m (+4.8%); ceilings 2.49 m (corner room) and 2.30 m (plain-wall room, closer
to typical than Depth Pro's 2.10 m); fireplace back wall 4.74 m, the same as Depth Pro. MoGe-2 is at least as
accurate here and is commercially licensed, so it is the recommended default (`depth_moge` in the cache).
The rendered images in `out/` still use Depth Pro depth.

**What this shows**
- Depth Pro is within ±5% on both rooms with a visible door, so it's the primary model.
  Depth Anything V2 reads 11–15% high and produced one impossible 24 m wall.
- That accuracy isn't guaranteed. Room 1's ceiling reads 2.10 m, which is lower than usual and may be a
  ~10% under-estimate. That's why the product needs a fallback.
- **Mitigation already built in:** when a door is visible, the pipeline computes a correction factor
  (room3 ×1.046, room5 ×0.972) and can re-render at corrected scale
  (`after_grid_door_calibrated.jpg`). In the product, the user would confirm "this is a door" or type
  one wall width. Disagreement between the two depth models is also a cheap confidence signal for
  deciding when to ask.

## Outputs (`out/`)

For each room: `before.jpg`, `after_wallpaper.jpg`, `after_tiles.jpg`, `after_grid.jpg`
(a 50 cm measurement grid, the visual scale check), `report.json`, and at the top level
`<room>__before_after.jpg` and `<room>__debug.jpg` (planes, depth, grid, calibrated grid).

## Known limitations (honest list)

1. **Mask edges** come from Mask2Former's ~1/4-resolution masks: they're slightly soft and ragged at
   some ceiling lines and next to furniture. Next step: refine edges with SAM 2, prompted by the wall mask.
2. **Lighting on dark or mixed materials.** Room 5 has light plaster and dark brick on one wall. A pretrained
   intrinsic decomposition model separates surface colour from lighting, and its albedo brightens the
   shading where the old wall is darker, so brick isn't read as shadow. Using its shading layer directly
   was worse: it moved real sunlight (room 4) into albedo. So the correction is one-way (it only brightens)
   and the sunlight survives. Residual: a slightly dark band remains under room 5's right-hand window.
3. **Small wall objects** such as sockets, switches and thermostats get papered over. A real install cuts
   around them, which needs a small detector or SAM 2 point prompts.
4. **Scale** is ±5% at best from a single photo, and worse on some photos. It needs the one-tap reference fallback.
5. **Products** here are procedural stand-ins with exact known size. Real catalogue images need a known
   DPI or repeat size per SKU, plus a seamless-tile check.
6. Mask2Former logs a "missing swin.layernorm" warning when loading. That layer isn't used by
   the segmentation head; the masks are as expected.

## Model licences (check before commercial use)

| Model | Licence on Hugging Face |
|---|---|
| Apple Depth Pro | metadata says `apple-amlr`; model card text says Apple-ASCL, so confirm with Apple's terms |
| Depth Anything V2 Large | CC BY-NC 4.0 (non-commercial); the Small model is Apache 2.0 |
| Mask2Former Swin-L ADE20K | "other", so check the model card |
| SAM 2.1 (planned) | Apache 2.0 |
| MoGe-2 (recommended metric depth) | MIT |
| compphoto/Intrinsic | check the repository licence |

The room photos are CC-licensed (see `data/rooms/CREDITS.md`). The rendered images in `out/` and `docs/` are
derivative works under the same licences.

## Run it

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m wallproof.infer      # models -> cache/*.npz (downloads ~5 GB of weights the first time)
python -m wallproof.run        # geometry + rendering -> out/
```

Code: `wallproof/infer.py` (models), `geometry.py` (planes, measurements), `render.py`
(texture mapping, lighting, compositing), `products.py` (product textures), `run.py` (outputs).
