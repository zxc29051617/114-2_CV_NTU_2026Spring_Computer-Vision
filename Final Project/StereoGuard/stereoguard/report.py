"""Save the result images, point cloud and a JSON / CSV summary."""
import csv
import json
import os

import cv2
import numpy as np

from . import viz
from .freespace import backproject


def summary_dict(res):
    objs = []
    for o in res.objects:
        objs.append({
            "name": o.det.name, "track_id": o.det.track_id, "confidence": round(o.det.conf, 3),
            "distance_m": round(o.dist_m, 3), "median_depth_m": round(o.median_m, 3),
            "lateral_m": round(o.x_m, 3), "height_m": None if np.isnan(o.height_m) else round(o.height_m, 3),
            "width_m": None if np.isnan(o.width_m) else round(o.width_m, 3),
            "reliable_ratio": round(o.valid_ratio, 3), "in_path": bool(o.in_path), "level": o.level,
        })
    fs = res.fs
    return {
        "intrinsics": res.intrinsics_source,
        "focal_px": round(res.f, 2),
        "image_size": [int(res.image.shape[1]), int(res.image.shape[0])],
        "baseline_cm": round(100 * res.baseline_m, 2),
        "baseline_source": res.baseline_source,
        "baseline_spread": None if np.isnan(res.baseline_spread) else round(res.baseline_spread, 3),
        "reliable_stereo_ratio": round(float(res.confident[res.valid].mean()), 3),
        "rectification": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.rect_stats.items()},
        "profile": res.profile.name,
        "thresholds_m": [res.profile.danger_m, res.profile.caution_m],
        "support_plane": None if fs is None else {
            "camera_height_m": round(fs.camera_height_m, 3), "inlier_ratio": round(fs.plane_inlier_ratio, 3)},
        "suggested_heading_deg": None if fs is None else fs.best_heading_deg,
        "free_straight_m": None if fs is None else round(fs.straight_free_m, 3),
        "hint": viz.hint_text(res.objects, fs, res.profile),
        "objects": objs,
        "timings_s": {k: round(v, 2) for k, v in res.timings.items()},
        "notes": res.notes,
    }


def write_ply(path, pts, cols):
    pts = pts.astype(np.float32)
    cols = cols.astype(np.uint8)
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.empty(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = pts, cols
        f.write(rec.tobytes())


def flying_pixels(depth, rel=0.03):
    """Pixels on depth discontinuities (they float between foreground and background in 3D)."""
    z = np.nan_to_num(depth, nan=0.0).astype(np.float32)
    gx = np.abs(cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)) / 8
    gy = np.abs(cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)) / 8
    bad = np.maximum(gx, gy) > rel * np.maximum(z, 1e-6)
    return cv2.dilate(bad.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def point_cloud(res, max_depth=None, step=2):
    depth = res.depth.copy()
    depth[flying_pixels(depth)] = np.nan
    pts, u, v = backproject(depth, res.f, res.cx, res.cy, step)
    cols = res.image[::step, ::step][..., ::-1]
    ok = np.isfinite(pts[..., 2]) & (pts[..., 2] > 0)
    if max_depth:
        ok &= pts[..., 2] < max_depth
    return pts[ok], cols[ok]


def orbit_frames(res, n=24, size=(640, 480), yaw_range=35, pitch=22):
    """Frames of a slow orbit around the reconstructed scene (for a GIF)."""
    pts, cols = point_cloud(res, max_depth=res.profile.range_m * 1.5, step=1)
    c = np.median(pts, 0)
    ext = np.linalg.norm(np.percentile(pts, 95, 0) - np.percentile(pts, 5, 0))
    yaws = yaw_range * np.sin(np.linspace(0, 2 * np.pi, n, endpoint=False))
    return [viz.render_points(pts, cols[:, ::-1], yaw_deg=y, pitch_deg=pitch, size=size, center=c, dist=1.0 * ext)
            for y in yaws]


def save_outputs(res, out_dir, prefix=""):
    os.makedirs(out_dir, exist_ok=True)
    j = lambda n: os.path.join(out_dir, prefix + n)
    ar = viz.draw_ar(res.image, res.objects, res.fs, res.profile)
    cv2.imwrite(j("ar.png"), ar)
    depth_img = viz.colorize_depth(res.depth)
    cv2.imwrite(j("depth.png"), depth_img)
    d = res.disp.copy()
    d[~res.valid] = 0
    dn = (255 * np.clip(d / max(1e-6, np.percentile(d[d > 0], 99)), 0, 1)).astype(np.uint8)
    cv2.imwrite(j("disparity.png"), cv2.applyColorMap(dn, cv2.COLORMAP_TURBO))
    if res.fs is not None:
        cv2.imwrite(j("topview.png"), viz.draw_bev(res.fs, res.objects, res.profile))
    pts, cols = point_cloud(res, max_depth=res.profile.range_m * 1.5)
    write_ply(j("pointcloud.ply"), pts * np.array([1, -1, -1]), cols)
    c = np.median(pts, 0)
    ext = np.linalg.norm(np.percentile(pts, 95, 0) - np.percentile(pts, 5, 0))
    cloud = viz.render_points(pts, cols[:, ::-1], yaw_deg=-30, pitch_deg=25, size=(800, 600), center=c, dist=ext)
    cv2.imwrite(j("pointcloud.png"), cloud)
    s = summary_dict(res)
    with open(j("summary.json"), "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    with open(j("objects.csv"), "w", newline="", encoding="utf-8") as f:
        if s["objects"]:
            w = csv.DictWriter(f, fieldnames=list(s["objects"][0]))
            w.writeheader()
            w.writerows(s["objects"])
    return s


def dashboard(res, width=1600):
    """One picture with the AR view, depth, top view and point cloud."""
    ar = viz.draw_ar(res.image, res.objects, res.fs, res.profile)
    depth = viz.colorize_depth(res.depth)
    top = viz.hstack_pad([ar, depth])
    panels = []
    if res.fs is not None:
        panels.append(viz.draw_bev(res.fs, res.objects, res.profile))
    pts, cols = point_cloud(res, max_depth=res.profile.range_m * 1.5)
    c = np.median(pts, 0)
    ext = np.linalg.norm(np.percentile(pts, 95, 0) - np.percentile(pts, 5, 0))
    panels.append(viz.render_points(pts, cols[:, ::-1], yaw_deg=-30, pitch_deg=25, size=(800, 600), center=c, dist=ext))
    bottom = viz.hstack_pad(panels)
    return viz.vstack_pad([top, bottom], w=width)
