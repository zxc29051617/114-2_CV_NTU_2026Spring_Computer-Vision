"""Disparity -> metric depth, baseline estimation and stereo / monocular fusion."""
import numpy as np


def lr_consistency(disp_l, disp_r, thresh=1.5):
    """Left-right check: disp_l(x) should agree with disp_r(x - disp_l(x))."""
    h, w = disp_l.shape
    xs = np.arange(w)[None, :] - disp_l
    xi = np.clip(np.round(xs).astype(np.int64), 0, w - 1)
    back = np.take_along_axis(disp_r, xi, axis=1)
    return (xs >= 0) & (np.abs(disp_l - back) <= thresh)


def disparity_to_depth(disp, f, baseline, min_disp=0.5, doffs=0.0):
    """Z = f * B / (d + doffs) (meters if B is in meters).  Invalid pixels -> NaN.

    doffs is the difference of the principal points of the two rectified views (0 for our phone
    rectification, non-zero for some Middlebury calibrations)."""
    depth = np.full(disp.shape, np.nan, dtype=np.float32)
    ok = disp + doffs > min_disp
    depth[ok] = f * baseline / (disp[ok] + doffs)
    return depth


def estimate_baseline(disp, f, mono_depth, valid, z_range=(0.1, 20.0), doffs=0.0):
    """Estimate the unknown length of the phone slide from a metric monocular depth map.

    For every reliable pixel  B_i = Z_mono_i * d_i / f ; the robust median is returned together
    with the spread (MAD / median), which tells how well the two depth maps agree in shape.
    """
    m = valid & (disp + doffs > 1.0) & np.isfinite(mono_depth) & (mono_depth > z_range[0]) & (mono_depth < z_range[1])
    if m.sum() < 100:
        return None, None
    b = mono_depth[m] * (disp[m] + doffs) / f
    med = float(np.median(b))
    spread = float(np.median(np.abs(b - med)) / med)
    return med, spread


def align_mono_to_stereo(mono_depth, stereo_depth, valid):
    """Fit 1/Z_mono -> 1/Z_stereo with scale + shift (least squares in inverse depth)."""
    m = valid & np.isfinite(stereo_depth) & np.isfinite(mono_depth) & (mono_depth > 0) & (stereo_depth > 0)
    if m.sum() < 100:
        return mono_depth
    x = 1.0 / mono_depth[m]
    y = 1.0 / stereo_depth[m]
    A = np.stack([x, np.ones_like(x)], 1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    inv = a / np.maximum(mono_depth, 1e-6) + b
    out = np.full_like(mono_depth, np.nan)
    ok = inv > 1e-4
    out[ok] = 1.0 / inv[ok]
    return out


def fuse(stereo_depth, confident, mono_depth=None):
    """Use stereo depth where the left-right check passes, otherwise the aligned mono depth."""
    if mono_depth is None:
        return stereo_depth.copy(), confident.copy()
    aligned = align_mono_to_stereo(mono_depth, stereo_depth, confident)
    fused = np.where(confident & np.isfinite(stereo_depth), stereo_depth, aligned).astype(np.float32)
    return fused, np.isfinite(fused)
