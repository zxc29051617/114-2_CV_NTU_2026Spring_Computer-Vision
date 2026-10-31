"""Video mode: per-frame depth, multi-object tracking, closing speed and time-to-collision.

Depth source per frame
  * stereo video (left/right frames + focal length + baseline): Fast-FoundationStereo
  * single phone video: metric Video Depth Anything over the whole clip (field of view from VGGT or
    MoGe-2 on the first frames)

Each track keeps a short history of its distance Z and lateral offset X.  A robust (Theil-Sen) line
over the last `window_s` seconds gives the closing speed v = -dZ/dt and the lateral speed; then
    TTC = Z / v          (only when the object is getting closer)
An object is in the path if its lateral extent overlaps the corridor now, or will overlap it at
the predicted time of contact.  A second, depth-free TTC from the growth of the mask size
(tau = s / (ds/dt), Lee 1976) is reported as a cross-check.
"""
from collections import defaultdict, deque

import numpy as np

from . import fusion, freespace, objects


def theil_sen(t, y):
    """Robust slope and intercept at t = 0 (median of pairwise slopes)."""
    i, j = np.triu_indices(len(t), 1)
    dt = t[j] - t[i]
    ok = dt > 1e-6
    slope = float(np.median((y[j][ok] - y[i][ok]) / dt[ok]))
    return slope, float(np.median(y - slope * t))


class TrackHistory:
    def __init__(self, window_s=1.0, min_span_s=0.5, min_samples=5):
        self.window_s = window_s
        self.min_span_s = min_span_s
        self.min_samples = min_samples
        self.hist = defaultdict(lambda: deque(maxlen=64))

    def update(self, track_id, t, z, x, size):
        self.hist[track_id].append((t, z, x, size))

    def kinematics(self, track_id):
        last = self.hist[track_id][-1][0]
        h = [r for r in self.hist[track_id] if r[0] >= last - self.window_s]
        if len(h) < self.min_samples or h[-1][0] - h[0][0] < self.min_span_s:
            return None
        a = np.array(h)
        t = a[:, 0] - a[-1, 0]
        vz, z_now = theil_sen(t, a[:, 1])
        vx, _ = theil_sen(t, a[:, 2])
        # scale-expansion TTC from sqrt(mask area): tau = 1 / (d log s / dt)
        g, _ = theil_sen(t, np.log(np.maximum(a[:, 3], 1e-6)))
        ttc_scale = 1.0 / g if g > 1e-3 else float("inf")
        return {"z": z_now, "vz": vz, "vx": vx, "ttc_scale": float(ttc_scale)}


def update_objects(objs, hist: TrackHistory, t, profile):
    """Attach speed / TTC to the measured objects and re-classify them."""
    for o in objs:
        tid = o.det.track_id
        if tid < 0:
            continue
        size = float(np.sqrt(o.det.mask.sum()))
        hist.update(tid, t, o.dist_m, o.x_m, size)
        k = hist.kinematics(tid)
        if k is None:
            continue
        closing = -k["vz"]
        o.speed_mps = closing
        o.extra.update(k)
        if closing > 0.3:
            o.ttc_s = k["z"] / closing
            # lateral position at the predicted time of contact (only for clear sideways motion,
            # otherwise the noise of the lateral speed would move parked cars into the path)
            if abs(k["vx"]) > 0.5:
                x_lo = o.x_lo + k["vx"] * o.ttc_s
                x_hi = o.x_hi + k["vx"] * o.ttc_s
                o.in_path = o.in_path or (x_hi > -profile.half_width_m and x_lo < profile.half_width_m)
        o.level = objects.classify(o.dist_m, o.in_path, profile, o.ttc_s)
    return objs


def analyse_frame(left, right, f, cx, cy, baseline_m, models, profile, depth=None):
    """Depth + objects + free space for one video frame.

    Stereo video: Fast-FoundationStereo with a left-right check.  Single-camera video: `depth`
    comes from Video Depth Anything run over the whole clip (temporally consistent)."""
    if right is not None:
        disp, disp_r = models.stereo(left, right, with_right=True)
        conf = fusion.lr_consistency(disp, disp_r) & (disp > 0.5)
        depth = fusion.disparity_to_depth(disp, f, baseline_m)
        depth[~conf] = np.nan
    else:
        conf = np.isfinite(depth)
    fs = freespace.analyse(depth, f, cx, cy, profile)
    dets = models.det.track(left)
    objs = objects.measure_objects(dets, depth, f, cx, cy, profile, reliable=conf,
                                   plane=fs.plane if fs is not None else None)
    return depth, conf, objs, fs
