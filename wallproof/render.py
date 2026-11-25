"""Stage 3: lay the product on every wall plane at true size and composite it.

For each wall pixel we intersect its camera ray with the fitted wall plane, express the hit
point in wall coordinates (metres along the wall, metres above the floor) and look the
product texture up at that physical position. Perspective, scale and pattern alignment
across the wall therefore come from the geometry, not from a 2-D warp guess.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .geometry import Geometry
from .products import PPM, Product


def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _mip_levels(tile_lin, n=7):
    levels = [tile_lin.astype(np.float32)]
    for _ in range(n - 1):
        h, w = levels[-1].shape[:2]
        if min(h, w) < 8:
            break
        levels.append(cv2.resize(levels[-1], (max(1, round(w / 2)), max(1, round(h / 2))),
                                 interpolation=cv2.INTER_AREA))
    return levels


def wall_uv(g: Geometry, scale=1.0):
    """Per-pixel wall coordinates in metres: (along-wall, height above floor)."""
    s = np.full((g.H, g.W), np.nan, np.float32)
    h = np.full((g.H, g.W), np.nan, np.float32)
    for k, p in enumerate(g.planes):
        m = g.plane_id == k
        if not m.any():
            continue
        r = g.rays[m]
        P = r * (p.off / (r @ p.n))[:, None] * scale
        s[m] = P @ p.h_axis
        h[m] = P @ g.up - g.floor_off * scale
    return s, h


def sample_product(prod: Product, s, h, valid):
    """Trilinear-filtered texture lookup at physical wall coordinates."""
    th, tw = prod.tile.shape[:2]
    ppm_x, ppm_y = tw / prod.size_m[0], th / prod.size_m[1]
    tx = np.where(valid, s, 0) * ppm_x
    ty = -np.where(valid, h, 0) * ppm_y  # texture rows run downwards, height runs up
    tx = np.mod(tx, tw).astype(np.float32)
    ty = np.mod(ty, th).astype(np.float32)
    # texture footprint per screen pixel (from the unwrapped coords) -> mip level
    gx1, gy1 = np.gradient(np.where(valid, s, np.nan) * ppm_x)
    gx2, gy2 = np.gradient(np.where(valid, h, np.nan) * ppm_y)
    fp = np.sqrt(np.nanmax(np.stack([gx1 ** 2 + gx2 ** 2, gy1 ** 2 + gy2 ** 2]), 0))
    fp = np.where(np.isfinite(fp), fp, 1.0)
    fp = ndimage.median_filter(fp, 5)
    lvl = np.clip(np.log2(np.maximum(fp, 1e-3)) - 0.3, 0, None)
    levels = _mip_levels(srgb_to_lin(prod.tile))
    lvl = np.minimum(lvl, len(levels) - 1)
    out = np.zeros((*s.shape, 3), np.float32)
    lo = np.floor(lvl).astype(int)
    frac = (lvl - lo)[..., None]
    samples = []
    for L, tex in enumerate(levels):
        sy, sx = tex.shape[0] / th, tex.shape[1] / tw
        samples.append(cv2.remap(tex, tx * sx, ty * sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP))
    samples.append(samples[-1])
    for L in range(len(levels)):
        m = lo == L
        if m.any():
            out[m] = (samples[L][m] * (1 - frac[m]) + samples[L + 1][m] * frac[m])
    return out


def shading(img_lin, wall_mask, detail=0.6, sigma_frac=0.02, gamma=0.75):
    """Illumination estimate on the walls (luminance only, no intrinsic model yet).

    smooth part : masked Gaussian of wall luminance -> light falloff, soft shadows
    detail part : ratio to that smooth part -> sharp cast shadows; suppressed where the old
                  wall itself is textured (brick, stucco) so its texture does not print through
    gamma < 1   : compresses very dark walls, which are more often dark paint than deep shadow
    """
    L = img_lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    m = wall_mask.astype(np.float32)
    sig = sigma_frac * max(L.shape)
    num = cv2.GaussianBlur(L * m, (0, 0), sig)
    den = cv2.GaussianBlur(m, (0, 0), sig)
    Ls = num / np.maximum(den, 1e-4)
    ref = np.percentile(L[wall_mask], 92)
    logL = np.log(L + 1e-3)
    hf = np.abs(logL - cv2.GaussianBlur(logL, (0, 0), 2.0))
    texture = cv2.GaussianBlur(hf * m, (0, 0), 12) / np.maximum(cv2.GaussianBlur(m, (0, 0), 12), 1e-4)
    w = detail * np.clip(1 - (texture - 0.04) / 0.08, 0, 1)
    D = np.clip(L / np.maximum(Ls, 1e-4), 0.55, 1.6) ** w
    S = (Ls / ref) ** gamma * D
    return np.clip(S, 0, 1.25).astype(np.float32)


def skirting_mask(img_lin, g: Geometry, h, labels, floor_ids=(3, 28), band_m=0.13):
    """Keep skirting boards. Per wall plane: find the wall/floor contact height, look at the
    band just above it, and keep that band only if it clearly differs from the wall above
    (a white board on a beige wall). Returns a mask of pixels to leave untouched."""
    keep = np.zeros(h.shape, bool)
    floor = np.isin(labels, floor_ids)
    near_floor = ndimage.binary_dilation(floor, iterations=3)
    for k in range(len(g.planes)):
        m = g.plane_id == k
        contact = m & near_floor
        if contact.sum() < 200:
            continue
        h0 = float(np.percentile(h[contact], 30))
        band = m & (h > h0 - 0.05) & (h < h0 + 0.07)
        above = m & (h > h0 + 0.25) & (h < h0 + 0.7)
        if band.sum() < 200 or above.sum() < 200:
            continue
        cb, ca = img_lin[band].mean(0), img_lin[above].mean(0)
        if np.abs(np.log((cb.mean() + 1e-3) / (ca.mean() + 1e-3))) < 0.25 and np.abs(cb / cb.sum() - ca / ca.sum()).sum() < 0.03:
            continue  # no visible skirting
        # board height: walk up the wall in 1 cm steps until the colour matches the wall above
        core = m & (h > h0 - 0.01) & (h < h0 + 0.03)
        if core.sum() < 100:
            continue
        cb = img_lin[core].mean(0)
        top = None
        for z in np.arange(0.02, band_m + 0.001, 0.01):
            sl = m & (h >= h0 + z) & (h < h0 + z + 0.01)
            if sl.sum() < 50:
                continue
            c = img_lin[sl].mean(0)
            if np.linalg.norm(c - ca) < np.linalg.norm(c - cb):
                top = z
                break
        if top is None:
            continue
        k_mask = m & (h < h0 + top + 0.005)
        keep |= k_mask
    return keep


def light_colour(img_lin, wall_mask, sigma_frac=0.04):
    """Spatial variation of the light colour on the walls (warm lamp glow vs cool daylight),
    relative to the wall's average colour, so the paint colour itself is not transferred."""
    m = wall_mask.astype(np.float32)
    sig = sigma_frac * max(wall_mask.shape)
    c = cv2.GaussianBlur(img_lin * m[..., None], (0, 0), sig) / np.maximum(cv2.GaussianBlur(m, (0, 0), sig), 1e-4)[..., None]
    chroma = c / np.maximum(c.mean(-1, keepdims=True), 1e-4)
    mean_chroma = chroma[wall_mask].mean(0)
    return np.clip(chroma / mean_chroma, 0.8, 1.25).astype(np.float32)


def shading_from_intrinsic(shd_lin, wall_mask):
    """Lighting from the pretrained intrinsic decomposition (image = albedo x shading).
    Returns grey shading normalised on the walls and the relative light colour."""
    shd = cv2.GaussianBlur(shd_lin.astype(np.float32), (0, 0), 1.2)
    L = shd @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    ref = np.percentile(L[wall_mask], 92)
    S = np.clip(L / max(ref, 1e-4), 0, 1.25)
    chroma = shd / np.maximum(shd.mean(-1, keepdims=True), 1e-4)
    col = np.clip(chroma / chroma[wall_mask].mean(0), 0.8, 1.25)
    return S.astype(np.float32), col.astype(np.float32)


def albedo_correction(albedo_lin, g: Geometry, wall_mask, sigma_frac=0.012):
    """One-way fix from the pretrained intrinsic model's albedo layer.

    Luminance-based shading reads a dark material (brick, dark paint) as shadow. Where the
    intrinsic albedo says this part of a wall is darker than the rest of that wall, brighten
    the shading by that ratio. Never darken: sunlight that the model files under albedo is
    left alone, so real sun patches and cast shadows survive."""
    H, W = wall_mask.shape
    A = cv2.resize(albedo_lin.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    A = A @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    m = wall_mask.astype(np.float32)
    sig = sigma_frac * max(H, W)
    As = cv2.GaussianBlur(A * m, (0, 0), sig) / np.maximum(cv2.GaussianBlur(m, (0, 0), sig), 1e-4)
    corr = np.ones((H, W), np.float32)
    for k in range(len(g.planes)):
        pm = (g.plane_id == k) & wall_mask
        if pm.sum() < 500:
            continue
        ref = np.percentile(As[pm], 60)
        corr[pm] = np.clip(ref / np.maximum(As[pm], 1e-4), 1.0, 3.0)
    return corr


def composite(img_rgb_u8, g: Geometry, wall_prob, prod: Product, scale=1.0, detail=0.6, labels=None,
              albedo_lin=None):
    img = img_rgb_u8.astype(np.float32) / 255
    img_lin = srgb_to_lin(img)
    wall = g.plane_id >= 0
    s, h = wall_uv(g, scale)
    if labels is not None:
        wall = wall & ~skirting_mask(img_lin, g, h, labels)
    tex = sample_product(prod, s, h, wall)
    S, col = shading(img_lin, wall, detail=detail), light_colour(img_lin, wall)
    if albedo_lin is not None:
        S = np.clip(S * albedo_correction(albedo_lin, g, wall), 0, 1.25)
    new_lin = tex * S[..., None] * col
    # soft alpha from the segmentation probability, restricted to assigned wall pixels
    a = np.clip((wall_prob.astype(np.float32) - 0.35) / 0.3, 0, 1)
    a = np.where(ndimage.binary_dilation(wall, iterations=2), a, 0)
    a = cv2.GaussianBlur(a, (0, 0), 0.8)[..., None]
    out = lin_to_srgb(new_lin * a + img_lin * (1 - a))
    return (out * 255 + 0.5).astype(np.uint8)
