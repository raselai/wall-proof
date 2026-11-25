"""Stage 2: room geometry from the cached model outputs.

floor plane -> "up" direction + floor height
walls       -> vertical planes (2-point RANSAC constrained to be vertical)
per pixel   -> which wall plane it belongs to
measure     -> wall sizes, door height, ceiling height (for the scale check)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

# ADE20K class ids (Mask2Former ADE semantic label space)
WALL, FLOOR, CEILING, WINDOW, DOOR, RUG = 0, 3, 5, 8, 14, 28
DOOR_IDS = (DOOR, 58)  # 14 door, 58 screen door / double door


@dataclass
class Plane:
    n: np.ndarray  # unit normal (camera coords), facing the camera
    off: float  # n . P = off for points P on the plane
    h_axis: np.ndarray  # unit horizontal in-plane axis, pointing image-right
    inliers: int = 0


@dataclass
class Geometry:
    H: int
    W: int
    f: float
    cx: float
    cy: float
    up: np.ndarray
    floor_off: float  # up . P = floor_off on the floor
    planes: list[Plane]
    plane_id: np.ndarray  # (H, W) int8, -1 = not wall
    rays: np.ndarray  # (H, W, 3)
    notes: list[str] = field(default_factory=list)


def rays_for(H, W, f, cx, cy):
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    return np.stack([(u - cx) / f, (v - cy) / f, np.ones_like(u)], -1)


def _fit_plane_lsq(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    return n, float(n @ c)


def fit_floor(P, rng):
    """RANSAC plane on floor points; returns up (unit) and offset."""
    best, best_n, best_off = 0, None, None
    cam_up = np.array([0, -1, 0], np.float32)
    for _ in range(300):
        idx = rng.choice(len(P), 3, replace=False)
        a, b, c = P[idx]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n @ cam_up < 0:
            n = -n
        if n @ cam_up < np.cos(np.deg2rad(35)):  # floor must be roughly horizontal
            continue
        off = n @ a
        thr = 0.02 + 0.01 * P[:, 2]
        cnt = int((np.abs(P @ n - off) < thr).sum())
        if cnt > best:
            best, best_n, best_off = cnt, n, off
    if best_n is None:
        return None
    inl = np.abs(P @ best_n - best_off) < 0.02 + 0.01 * P[:, 2]
    n, off = _fit_plane_lsq(P[inl])
    if n @ cam_up < 0:
        n, off = -n, -off
    return n.astype(np.float32), float(off), inl.mean()


def ransac_vertical_planes(P, up, rng, max_planes=5, min_frac=0.04, iters=400):
    """Vertical planes: normal must be perpendicular to `up`. 2-point minimal sample."""
    planes = []
    remaining = np.ones(len(P), bool)
    total = len(P)
    # Remove the up component so all fitting happens in the horizontal plane.
    Ph = P - np.outer(P @ up, up)
    thr_all = 0.03 + 0.015 * P[:, 2]
    for _ in range(max_planes):
        idx_rem = np.flatnonzero(remaining)
        if len(idx_rem) < min_frac * total:
            break
        best, best_n, best_off = 0, None, None
        for _ in range(iters):
            i, j = rng.choice(idx_rem, 2, replace=False)
            d = Ph[j] - Ph[i]
            if np.linalg.norm(d) < 0.2:
                continue
            n = np.cross(up, d)
            n /= np.linalg.norm(n)
            off = n @ P[i]
            cnt = int((np.abs(P[idx_rem] @ n - off) < thr_all[idx_rem]).sum())
            if cnt > best:
                best, best_n, best_off = cnt, n, off
        if best_n is None or best < min_frac * total:
            break
        # refine: 2-D line fit in the horizontal plane (keeps the plane exactly vertical)
        inl = idx_rem[np.abs(P[idx_rem] @ best_n - best_off) < thr_all[idx_rem]]
        Q = Ph[inl]
        c = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        dirn = vt[0] - (vt[0] @ up) * up
        n = np.cross(up, dirn / np.linalg.norm(dirn))
        n /= np.linalg.norm(n)
        off = float(n @ c)
        if off > 0:  # face the camera: camera origin on the negative side
            n, off = -n, -off
        inl = idx_rem[np.abs(P[idx_rem] @ n - off) < thr_all[idx_rem]]
        h_axis = np.cross(up, n)
        if h_axis[0] < 0:
            h_axis = -h_axis
        planes.append(Plane(n.astype(np.float32), off, h_axis.astype(np.float32), len(inl)))
        remaining[inl] = False
    return planes


def _refit_vertical(P, up):
    Ph = P - np.outer(P @ up, up)
    c = Ph.mean(0)
    _, _, vt = np.linalg.svd(Ph - c, full_matrices=False)
    dirn = vt[0] - (vt[0] @ up) * up
    n = np.cross(up, dirn / np.linalg.norm(dirn))
    n /= np.linalg.norm(n)
    off = float(n @ c)
    if off > 0:
        n, off = -n, -off
    h_axis = np.cross(up, n)
    if h_axis[0] < 0:
        h_axis = -h_axis
    return n.astype(np.float32), off, h_axis.astype(np.float32)


def _merge_similar(planes, P, up, ang_deg=10, rel_dist=0.06):
    """Monocular depth bends large flat walls slightly, so RANSAC can split one wall into two
    near-parallel planes. Merge planes whose normals agree and whose offsets differ by less
    than `rel_dist` of their distance, then refit on the union of their points."""
    planes = sorted(planes, key=lambda p: -p.inliers)
    changed = True
    while changed:
        changed = False
        for i in range(len(planes)):
            for j in range(i + 1, len(planes)):
                a, b = planes[i], planes[j]
                ang = np.degrees(np.arccos(np.clip(a.n @ b.n, -1, 1)))
                if ang < ang_deg and abs(a.off - b.off) < rel_dist * max(abs(a.off), abs(b.off)):
                    thr = 0.03 + 0.015 * P[:, 2]
                    m = (np.abs(P @ a.n - a.off) < thr) | (np.abs(P @ b.n - b.off) < thr)
                    n, off, h = _refit_vertical(P[m], up)
                    planes[i] = Plane(n, off, h, int(m.sum()))
                    del planes[j]
                    changed = True
                    break
            if changed:
                break
    return planes


def build(labels, wall_prob, depth, f, seed=0) -> Geometry:
    rng = np.random.default_rng(seed)
    H, W = depth.shape
    cx, cy = (W - 1) / 2, (H - 1) / 2
    rays = rays_for(H, W, f, cx, cy)
    P = rays * depth[..., None]
    notes = []

    floor_mask = np.isin(labels, (FLOOR, RUG))
    fm = np.flatnonzero(floor_mask.ravel())
    up = np.array([0, -1, 0], np.float32)
    floor_off = None
    if len(fm) > 2000:
        sel = rng.choice(fm, min(20000, len(fm)), replace=False)
        r = fit_floor(P.reshape(-1, 3)[sel], rng)
        if r is not None:
            up, floor_off, frac = r
            tilt = np.degrees(np.arccos(np.clip(up @ np.array([0, -1, 0]), -1, 1)))
            notes.append(f"floor plane found ({frac:.0%} inliers), camera tilt {tilt:.1f} deg")
    if floor_off is None:
        notes.append("no floor visible: assuming level camera, floor at lowest wall point")

    wall = (labels == WALL) & (wall_prob.astype(np.float32) > 0.5)
    wm = np.flatnonzero(wall.ravel())
    sel = rng.choice(wm, min(30000, len(wm)), replace=False)
    Pw = P.reshape(-1, 3)[sel]
    planes = _merge_similar(ransac_vertical_planes(Pw, up, rng), Pw, up)
    if floor_off is None:
        floor_off = float(np.percentile(Pw @ up, 1))

    # Per-pixel plane assignment: which plane best explains this pixel's depth.
    pid = np.full((H, W), -1, np.int8)
    if planes:
        Pf = P[wall]
        thr = 0.03 + 0.015 * Pf[:, 2]
        res = np.stack([np.abs(Pf @ p.n - p.off) / thr for p in planes], 1)
        # rays must hit the plane in front of the camera
        for k, p in enumerate(planes):
            res[(rays[wall] @ p.n) * p.off <= 0, k] = np.inf
        best = res.argmin(1)
        ok = res[np.arange(len(best)), best] < 4.0
        tmp = np.full(len(best), -1, np.int8)
        tmp[ok] = best[ok]
        pid[wall] = tmp
        # wall pixels no plane explains (depth noise at edges): take the nearest assigned label
        miss = wall & (pid < 0)
        if miss.any() and (pid >= 0).any():
            _, (iy, ix) = ndimage.distance_transform_edt(pid < 0, return_indices=True)
            pid[miss] = pid[iy[miss], ix[miss]]
        # clean speckle: majority filter inside the wall region
        pid = _mode_filter(pid, wall, len(planes), size=21)
        # drop tiny islands
        for k in range(len(planes)):
            lab, n = ndimage.label(pid == k)
            if n > 1:
                sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
                for i, s in enumerate(sizes, 1):
                    if s < 0.002 * H * W:
                        pid[lab == i] = -1
        miss = wall & (pid < 0)
        if miss.any() and (pid >= 0).any():
            _, (iy, ix) = ndimage.distance_transform_edt(pid < 0, return_indices=True)
            pid[miss] = pid[iy[miss], ix[miss]]
    notes.append(f"{len(planes)} wall plane(s)")
    return Geometry(H, W, f, cx, cy, up, floor_off, planes, pid, rays, notes)


def _mode_filter(pid, valid, k, size):
    votes = np.stack([ndimage.uniform_filter((pid == i).astype(np.float32), size) for i in range(k)], 0)
    out = pid.copy()
    out[valid] = votes.argmax(0)[valid].astype(np.int8)
    return out


def plane_points(g: Geometry, k: int, mask=None):
    """3-D points (camera coords) where pixel rays hit plane k."""
    p = g.planes[k]
    m = (g.plane_id == k) if mask is None else mask
    r = g.rays[m]
    t = p.off / (r @ p.n)
    return r * t[:, None]


def measure(g: Geometry, labels, depth):
    """Real-world measurements used as a scale sanity check."""
    out = {"walls": []}
    P = g.rays * depth[..., None]
    height = P @ g.up - g.floor_off
    for k, p in enumerate(g.planes):
        Q = plane_points(g, k)
        if len(Q) < 500:
            continue
        s = Q @ p.h_axis
        hgt = Q @ g.up - g.floor_off
        out["walls"].append(dict(
            plane=k, visible_width_m=float(np.percentile(s, 99) - np.percentile(s, 1)),
            visible_top_m=float(np.percentile(hgt, 99.5)), distance_m=float(abs(p.off))))
    ceil = labels == CEILING
    if ceil.sum() > 0.01 * g.H * g.W:
        out["ceiling_height_m"] = float(np.median(height[ceil]))
    # Door = largest door-labelled region that is door-shaped and stands on the floor.
    door = np.isin(labels, DOOR_IDS)
    lab, n = ndimage.label(door)
    for sl, i in sorted(((s, i) for i, s in enumerate(ndimage.find_objects(lab), 1)),
                        key=lambda t: -(lab[t[0]] == t[1]).sum()):
        m = lab == i
        hpx, wpx = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        bottom = height[m][np.nonzero(m)[0] >= np.percentile(np.nonzero(m)[0], 98)]
        on_floor = abs(float(np.median(bottom))) < 0.25
        if m.sum() < 0.004 * g.H * g.W or hpx < 1.4 * wpx or not on_floor:
            continue
        ys = np.nonzero(m)[0]
        top_rows = m & (np.arange(g.H)[:, None] <= np.percentile(ys, 2))
        out["door"] = dict(top_m=float(np.median(height[top_rows])), pixels=int(m.sum()), touches_floor=True)
        break
    return out
