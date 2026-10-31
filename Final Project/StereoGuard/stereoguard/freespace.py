"""Ground plane, top-down occupancy grid and a walkable-direction hint.

1. back-project the depth map to a point cloud (camera frame: X right, Y down, Z forward)
2. RANSAC a support plane (floor / table top) whose normal is close to the camera "up" direction
3. points standing 'h_min'..'h_max' above that plane are obstacles; points on it are free ground
4. obstacles go into a bird's-eye occupancy grid (each pixel weighted by the surface area it covers)
5. for every heading inside the field of view we sweep a corridor as wide as the user/robot
   and record the distance to the first occupied cell (a polar histogram, as in the Vector Field
   Histogram of Borenstein & Koren); the suggested heading maximises free distance with a small
   penalty for turning.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Profile:
    name: str
    danger_m: float        # in-path objects closer than this -> DANGER
    caution_m: float       # in-path objects closer than this -> CAUTION
    half_width_m: float    # half width of the swept corridor (body / robot)
    h_min: float           # obstacle height band above the support plane
    h_max: float
    range_m: float         # radar / BEV range
    cell_m: float          # BEV cell size
    ttc_danger_s: float = 1.5
    ttc_caution_s: float = 3.0


PROFILES = {
    "tabletop": Profile("tabletop", 0.40, 0.80, 0.06, 0.015, 0.60, 1.6, 0.01),
    "walking": Profile("walking", 1.00, 2.00, 0.35, 0.10, 2.00, 6.0, 0.05),
    "driving": Profile("driving", 8.00, 15.0, 0.95, 0.30, 3.00, 40.0, 0.20, 2.0, 4.0),
}


def backproject(depth, f, cx, cy, step=1):
    h, w = depth.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth[::step, ::step]
    x = (u - cx) * z / f
    y = (v - cy) * z / f
    return np.stack([x, y, z], -1), u, v


@dataclass
class Plane:
    n: np.ndarray   # unit normal pointing up (towards the camera side)
    d: float        # height(p) = n.p + d ; camera height = d

    def height(self, p):
        return p @ self.n + self.d


def fit_support_plane(depth, f, cx, cy, max_tilt_deg=40.0, iters=400, step=4, rng=None, z_max=None):
    """RANSAC for the dominant roughly-horizontal plane below the camera."""
    rng = rng or np.random.default_rng(0)
    pts, u, v = backproject(depth, f, cx, cy, step)
    ok = np.isfinite(pts[..., 2]) & (pts[..., 2] > 0)
    if z_max:
        ok &= pts[..., 2] < z_max
    ok &= v > cy - 0.1 * depth.shape[0]          # support surface is (mostly) below the horizon
    P = pts[ok]
    if len(P) < 200:
        return None, 0.0
    up = np.array([0.0, -1.0, 0.0])
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    best, best_score = None, 0
    zs = P[:, 2]
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if n @ up < 0:
            n = -n
        if n @ up < cos_tilt:
            continue
        d = -n @ s[0]
        if d <= 0:                      # the camera has to be above the plane
            continue
        tol = 0.01 + 0.01 * zs          # tolerance grows with depth (stereo error ~ Z^2)
        score = np.sum(np.abs(P @ n + d) < tol)
        if score > best_score:
            best, best_score = (n, d), score
    if best is None:
        return None, 0.0
    # least-squares refinement on the inliers
    n, d = best
    inl = np.abs(P @ n + d) < (0.01 + 0.01 * zs)
    Q = P[inl]
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    n2 = vt[2] if vt[2] @ up > 0 else -vt[2]
    d2 = -n2 @ c
    if d2 > 0 and n2 @ up >= cos_tilt:
        n, d = n2, d2
    return Plane(n, float(d)), float(inl.mean())


def ground_frame(plane: Plane):
    """Unit vectors (right, forward, up) of a frame lying on the support plane."""
    up = plane.n
    z = np.array([0.0, 0.0, 1.0])
    fwd = z - (z @ up) * up
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    return right, fwd, up


@dataclass
class FreeSpace:
    grid_occ: np.ndarray      # (nz, nx) bool, row 0 = far
    grid_free: np.ndarray     # (nz, nx) bool, observed ground
    extent: tuple             # (x_min, x_max, z_min, z_max) in meters
    headings_deg: np.ndarray
    free_dist: np.ndarray     # free distance per heading (m)
    best_heading_deg: float
    straight_free_m: float
    ground_mask: np.ndarray   # (H, W) bool: pixels on the support plane
    obstacle_mask: np.ndarray  # (H, W) bool
    plane: Plane
    plane_inlier_ratio: float
    camera_height_m: float


def analyse(depth, f, cx, cy, profile: Profile, hfov_deg=None, plane=None):
    h, w = depth.shape
    if plane is None:
        plane, ratio = fit_support_plane(depth, f, cx, cy, z_max=profile.range_m * 1.5)
    else:
        ratio = float("nan")
    if plane is None:
        return None
    pts, _, _ = backproject(depth, f, cx, cy, 1)
    valid = np.isfinite(pts[..., 2]) & (pts[..., 2] > 0)
    P = np.where(valid[..., None], pts, 0.0)
    right, fwd, up = ground_frame(plane)
    hgt = P @ up + plane.d
    xg = P @ right
    zg = P @ fwd
    tol = 0.01 + 0.01 * np.nan_to_num(pts[..., 2])
    ground = valid & (np.abs(hgt) < tol)
    obst = valid & (hgt > profile.h_min) & (hgt < profile.h_max) & (zg > 0)

    cell = profile.cell_m
    x_half = profile.range_m * 0.6
    nx = int(np.ceil(2 * x_half / cell))
    nz = int(np.ceil(profile.range_m / cell))
    extent = (-x_half, x_half, 0.0, profile.range_m)

    def to_cells(mask):
        ix = np.floor((xg[mask] + x_half) / cell).astype(int)
        iz = np.floor(zg[mask] / cell).astype(int)
        keep = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
        return ix[keep], iz[keep], keep

    # obstacle evidence: surface area covered by each pixel ~ (Z / f)^2
    ix, iz, keep = to_cells(obst)
    area = (pts[..., 2][obst][keep] / f) ** 2
    acc = np.zeros((nz, nx))
    np.add.at(acc, (iz, ix), area)
    occ = acc >= 0.25 * cell * cell
    gx, gz, _ = to_cells(ground)
    free = np.zeros((nz, nx), bool)
    free[gz, gx] = True
    free &= ~occ

    # polar free-distance histogram: march along each heading with a corridor as wide as the
    # user / robot; stop at the first occupied cell, and do not trust ground that was never seen
    # (a table edge or a drop-off ends the observed ground as well)
    import cv2
    half = (hfov_deg or np.degrees(2 * np.arctan(w / (2 * f)))) / 2
    headings = np.arange(-np.floor(half), np.floor(half) + 0.1, 2.0)
    r = int(np.ceil(profile.half_width_m / cell))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    occ_d = cv2.dilate(occ.astype(np.uint8), disk).astype(bool)
    seen = cv2.dilate(free.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) | occ
    steps = np.arange(cell / 2, profile.range_m, cell / 2)
    max_gap = max(4 * cell, 0.08 * profile.range_m)
    gz_, gx_ = np.nonzero(free)
    near_m = float(np.percentile((gz_ + 0.5) * cell, 2)) if gz_.size else 0.0   # closest visible ground
    free_dist = np.zeros(len(headings))
    for i, th in enumerate(np.radians(headings)):
        xs, zs = steps * np.sin(th), steps * np.cos(th)
        ix = np.floor((xs + x_half) / cell).astype(int)
        iz = np.floor(zs / cell).astype(int)
        inside = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
        stop = len(steps)
        hit = np.nonzero(inside & occ_d[np.clip(iz, 0, nz - 1), np.clip(ix, 0, nx - 1)])[0]
        if hit.size:
            stop = hit[0]
        seen_ray = inside & seen[np.clip(iz, 0, nz - 1), np.clip(ix, 0, nx - 1)]
        first = np.nonzero(seen_ray[:stop])[0]
        if first.size == 0:
            # no ground seen on this heading: only the part below the image (up to the obstacle) is known
            free_dist[i] = steps[stop - 1] if 0 < stop < len(steps) and steps[stop - 1] < near_m else 0.0
            continue
        # the ground right in front of the camera is below the image; it counts as free
        last, gap_start = steps[first[0]], None
        for k in range(first[0], stop):
            if seen_ray[k]:
                last, gap_start = steps[k], None
            else:
                gap_start = steps[k] if gap_start is None else gap_start
                if steps[k] - gap_start > max_gap:
                    break
        free_dist[i] = steps[stop - 1] if (stop < len(steps) and steps[stop - 1] - last <= max_gap) else last
    score = free_dist - 0.1 * free_dist.max() * np.abs(headings) / max(1.0, half)
    best = float(headings[int(np.argmax(score))])
    straight = float(free_dist[int(np.argmin(np.abs(headings)))])
    return FreeSpace(occ, free, extent, headings, free_dist, best, straight, ground, obst, plane, ratio,
                     float(plane.d))
