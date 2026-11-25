"""Product textures with known physical size.

In production these come from the shop catalogue: a scanned/flat product image plus its
real repeat size (e.g. "roll width 53 cm, pattern repeat 53 cm"). Here we generate
seamless stand-ins procedurally so the physical size is exact and known.

Every product is a seamless RGB tile (float32, linear 0..1) + its size in metres.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PPM = 1000  # texture resolution: pixels per metre (1 px = 1 mm)


@dataclass
class Product:
    key: str
    name: str
    tile: np.ndarray  # (h, w, 3) float32 sRGB 0..1, seamless
    size_m: tuple[float, float]  # (width, height) of one tile in metres
    note: str


def _srgb(hex_: str) -> np.ndarray:
    return np.array([int(hex_[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


def deco_fan_wallpaper(ss: int = 3) -> Product:
    """Art-deco scallop fans. Roll width 53 cm, 2 fans across (26.5 cm each)."""
    w_m, h_m = 0.53, 0.53
    D = w_m / 2  # fan diameter
    r = D / 2
    step = r * 0.5 * 2  # vertical row step = r -> 4 rows per 53 cm repeat
    n = int(round(w_m * PPM)) * ss
    m = int(round(h_m * PPM)) * ss
    ys, xs = np.mgrid[0:m, 0:n].astype(np.float32) / (PPM * ss)
    bg, ring_a, ring_b, gold = _srgb("#123b3d"), _srgb("#174a4b"), _srgb("#1c5856"), _srgb("#c9a45c")
    out = np.empty((m, n, 3), np.float32)
    out[:] = bg
    rows = int(round(h_m / step))
    for k in range(-1, rows + 2):  # drawn top -> bottom, later rows on top
        cy = k * step
        off = (k % 2) * r
        for i in range(-1, 4):
            cx = i * D + off
            dx = (xs - cx + w_m / 2) % w_m - w_m / 2  # periodic in x
            dy = ys - cy
            dy = np.where(np.abs(dy) > h_m / 2, dy - np.sign(dy) * h_m, dy)  # periodic in y
            d = np.sqrt(dx * dx + dy * dy)
            inside = (d < r) & (dy <= 0)
            if not inside.any():
                continue
            t = d / r  # 0 centre .. 1 rim
            band = np.floor(t * 5).astype(int)
            col = np.where((band % 2 == 0)[..., None], ring_a, ring_b)
            line = np.abs(t * 5 - np.round(t * 5)) < 0.06  # concentric gold arcs
            rim = t > 0.965
            ang = np.arctan2(-dy, dx)  # 0..pi on upper half
            ray = (np.abs(((ang / np.pi) * 12) - np.round((ang / np.pi) * 12)) < 0.05) & (t < 0.4) & (t > 0.08)
            col = np.where((line | rim | ray)[..., None], gold, col)
            out[inside] = col[inside]
    tile = cv2.resize(out, (n // ss, m // ss), interpolation=cv2.INTER_AREA)
    return Product("wallpaper", "Deco Fan wallpaper", tile, (w_m, h_m),
                   "roll width 53 cm, pattern repeat 53 cm, each fan 26.5 cm wide")


def subway_tiles(seed: int = 7) -> Product:
    """Gloss 30 x 10 cm tiles, running bond, 3 mm grout. 2.4 m x 1.2 m texture."""
    tw, th, g = 0.30, 0.10, 0.003
    w_m, h_m = 2.4, 1.2
    n, m = int(w_m * PPM), int(h_m * PPM)
    rng = np.random.default_rng(seed)
    base = _srgb("#e9e4da")
    grout = _srgb("#9c968c")
    out = np.empty((m, n, 3), np.float32)
    out[:] = grout
    ys, xs = np.mgrid[0:m, 0:n].astype(np.float32) / PPM
    row = np.floor(ys / th).astype(int)
    xo = xs + (row % 2) * tw / 2
    col = np.floor(xo / tw).astype(int) % int(round(w_m / tw))
    lx = xo % tw
    ly = ys % th
    in_tile = (lx > g / 2) & (lx < tw - g / 2) & (ly > g / 2) & (ly < th - g / 2)
    tone = rng.normal(0, 0.025, size=(int(round(h_m / th)), int(round(w_m / tw)) + 1))
    tint = tone[row % tone.shape[0], col]
    # soft bevel: darker towards the tile edge
    e = np.minimum(np.minimum(lx, tw - lx), np.minimum(ly, th - ly))
    bevel = np.clip(e / 0.006, 0, 1) ** 0.5
    tilecol = base[None, None] * (1 + tint[..., None]) * (0.82 + 0.18 * bevel[..., None])
    out[in_tile] = tilecol[in_tile]
    return Product("tiles", "Subway tile 30x10 cm", np.clip(out, 0, 1), (w_m, h_m),
                   "30 x 10 cm tiles, running bond, 3 mm grout, anchored to the floor line")


def scale_grid() -> Product:
    """Measurement texture: 50 cm checker squares with 10 cm tick lines."""
    w_m = h_m = 1.0
    n = m = int(w_m * PPM)
    ys, xs = np.mgrid[0:m, 0:n].astype(np.float32) / PPM
    a, b = _srgb("#f2efe6"), _srgb("#d8e3ec")
    chk = ((np.floor(xs / 0.5) + np.floor(ys / 0.5)) % 2).astype(bool)
    out = np.where(chk[..., None], b, a).astype(np.float32)
    minor = (np.minimum(xs % 0.1, 0.1 - xs % 0.1) < 0.0015) | (np.minimum(ys % 0.1, 0.1 - ys % 0.1) < 0.0015)
    major = (np.minimum(xs % 0.5, 0.5 - xs % 0.5) < 0.006) | (np.minimum(ys % 0.5, 0.5 - ys % 0.5) < 0.006)
    out[minor] = _srgb("#9fb0be")
    out[major] = _srgb("#c0392b")
    img = (out * 255).astype(np.uint8)
    for cx in (0.25, 0.75):
        for cy in (0.25, 0.75):
            txt = "50 cm"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 3.0, 6)
            cv2.putText(img, txt, (int(cx * PPM - tw / 2), int(cy * PPM + th / 2)), cv2.FONT_HERSHEY_DUPLEX,
                        3.0, (192, 57, 43), 6, cv2.LINE_AA)
    return Product("grid", "Scale check grid", img.astype(np.float32) / 255, (w_m, h_m),
                   "red lines every 50 cm, blue lines every 10 cm; floor-anchored")


def all_products() -> list[Product]:
    return [deco_fan_wallpaper(), subway_tiles(), scale_grid()]


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / "out" / "products"
    out.mkdir(parents=True, exist_ok=True)
    for p in all_products():
        reps = (2, 2) if p.key != "tiles" else (1, 1)
        img = np.tile(p.tile, (reps[1], reps[0], 1))
        cv2.imwrite(str(out / f"{p.key}.png"), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(p.key, p.tile.shape, p.size_m)
