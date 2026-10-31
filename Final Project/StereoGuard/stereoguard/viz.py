"""Drawing: AR overlay, depth colouring, bird's-eye radar and point-cloud rendering."""
import cv2
import numpy as np

from .objects import LEVEL_COLORS

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _text(img, txt, org, scale, color, thick=1, bg=None, pad=4):
    (tw, th), base = cv2.getTextSize(txt, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + base + pad // 2), bg, -1)
    cv2.putText(img, txt, (x, y), FONT, scale, color, thick, cv2.LINE_AA)
    return tw, th


def colorize_depth(depth, vmin=None, vmax=None, colorbar=True):
    """Turbo colouring of inverse depth (near = red, far = blue); NaN -> black."""
    ok = np.isfinite(depth) & (depth > 0)
    if vmin is None:
        vmin = float(np.percentile(depth[ok], 2)) if ok.any() else 0.1
    if vmax is None:
        vmax = float(np.percentile(depth[ok], 98)) if ok.any() else 10.0
    inv = np.zeros_like(depth, dtype=np.float32)
    inv[ok] = 1.0 / np.clip(depth[ok], vmin, vmax)
    t = np.clip((inv - 1.0 / vmax) / (1.0 / vmin - 1.0 / vmax + 1e-9), 0, 1)
    col = cv2.applyColorMap((t * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[~ok] = 0
    if not colorbar:
        return col
    h, w = col.shape[:2]
    bar_w = max(60, w // 12)
    bar = np.zeros((h, bar_w, 3), np.uint8) + 30
    y0, y1 = int(0.08 * h), int(0.92 * h)
    for y in range(y0, y1):
        tt = 1 - (y - y0) / (y1 - y0)          # top = near
        c = cv2.applyColorMap(np.uint8([[tt * 255]]), cv2.COLORMAP_TURBO)[0, 0]
        bar[y, 8:24] = c
    scale = max(0.35, w / 2200)
    for k in range(5):
        tt = 1 - k / 4
        z = 1.0 / (tt * (1.0 / vmin - 1.0 / vmax) + 1.0 / vmax)
        y = int(y0 + k / 4 * (y1 - y0))
        _text(bar, f"{z:.2f}m" if z < 10 else f"{z:.0f}m", (27, y + 4), scale, (235, 235, 235))
    return np.hstack([col, bar])


def draw_ar(img, objects, fs=None, profile=None, title="StereoGuard", show_ground=True, max_labels=6):
    """Annotated camera view: masks coloured by warning level, distances, walkable ground, heading."""
    out = img.copy()
    h, w = out.shape[:2]
    s = min(w, 1.6 * h) / 1000.0      # wide video frames get the same text size as a 1000 x 625 image
    if fs is not None and show_ground:
        g = fs.ground_mask
        tint = out.copy()
        tint[g] = (0.55 * tint[g] + 0.45 * np.array([90, 200, 90])).astype(np.uint8)
        out = tint
    overlay = out.copy()
    for o in objects[::-1]:
        col = LEVEL_COLORS[o.level]
        overlay[o.det.mask] = (0.5 * overlay[o.det.mask] + 0.5 * np.array(col)).astype(np.uint8)
    out = overlay
    for o in objects[::-1]:
        col = LEVEL_COLORS[o.level]
        cnts, _ = cv2.findContours(o.det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, col, max(2, int(2 * s)), cv2.LINE_AA)
    # labels last so they stay readable: warnings first, then the nearest objects; a label that
    # would overlap one already drawn is moved down a little or dropped
    order = {"DANGER": 0, "CAUTION": 1, "SAFE": 2}
    ranked = sorted(objects, key=lambda o: (order[o.level], o.dist_m))
    placed = []
    fs1 = 0.62 * s * 1.25
    thick = max(1, int(2 * s))
    pad = int(5 * s)
    for o in ranked[:max_labels]:
        col = LEVEL_COLORS[o.level]
        ys, xs = np.nonzero(o.det.mask)
        x0, y0 = int(np.percentile(xs, 5)), int(ys.min())
        name = o.det.name + (f" #{o.det.track_id}" if o.det.track_id >= 0 else "")
        line1 = f"{name} {o.dist_m:.2f} m"
        line2 = o.level + (" | in path" if o.in_path else "")
        if np.isfinite(o.ttc_s) and o.ttc_s < 30:
            line2 += f" | TTC {o.ttc_s:.1f}s"
        (tw1, th), _ = cv2.getTextSize(line1, FONT, fs1, thick)
        tw2 = cv2.getTextSize(line2, FONT, fs1 * 0.8, thick)[0][0]
        box_w, box_h = max(tw1, tw2) + 2 * pad, int(2.9 * th) + 3 * pad
        y_text = y0 - int(1.9 * th) - 6
        top_free = int(46 * s) + th + int(8 * s)      # keep labels below the header bar
        if y_text < top_free:
            y_text = max(y0 + th + 8, top_free)
        x0 = int(np.clip(x0, 6, w - box_w - 6))
        for _ in range(4):
            rect = (x0 - pad, y_text - th - pad, x0 - pad + box_w, y_text - th - pad + box_h)
            if not any(r[0] < rect[2] and rect[0] < r[2] and r[1] < rect[3] and rect[1] < r[3] for r in placed):
                break
            y_text += box_h
        else:
            continue
        if rect[3] > h:
            continue
        placed.append(rect)
        _text(out, line1, (x0, y_text), fs1, (255, 255, 255), thick, bg=(20, 20, 20), pad=pad)
        _text(out, line2, (x0, y_text + int(1.6 * th) + 4), fs1 * 0.8, (20, 20, 20), thick, bg=col, pad=pad)
    # heading arrow
    if fs is not None:
        base = (w // 2, h - int(18 * s))
        L = int(120 * s)
        th = np.radians(fs.best_heading_deg)
        tip = (int(base[0] + L * np.sin(th)), int(base[1] - L * np.cos(th)))
        cv2.arrowedLine(out, base, tip, (255, 255, 255), int(10 * s), cv2.LINE_AA, tipLength=0.3)
        cv2.arrowedLine(out, base, tip, (230, 120, 20), int(6 * s), cv2.LINE_AA, tipLength=0.3)
    # header
    bar_h = int(46 * s)
    cv2.rectangle(out, (0, 0), (w, bar_h), (25, 25, 25), -1)
    _text(out, title, (int(12 * s), int(31 * s)), 0.8 * s, (255, 255, 255), max(1, int(2 * s)))
    summary = hint_text(objects, fs, profile)
    tw = cv2.getTextSize(summary, FONT, 0.62 * s, max(1, int(2 * s)))[0][0]
    _text(out, summary, (w - tw - int(12 * s), int(30 * s)), 0.62 * s, (200, 235, 255), max(1, int(2 * s)))
    return out


def hint_text(objects, fs, profile):
    in_path = [o for o in objects if o.in_path]
    parts = []
    if in_path:
        o = in_path[0]
        parts.append(f"ahead: {o.det.name} {o.dist_m:.2f} m [{o.level}]")
    else:
        parts.append("nothing in path")
    if fs is not None:
        b = fs.best_heading_deg
        if abs(b) < 3:
            parts.append(f"go straight ({fs.straight_free_m:.2f} m free)")
        else:
            parts.append(f"veer {'right' if b > 0 else 'left'} {abs(b):.0f} deg")
    return " | ".join(parts)


def draw_bev(fs, objects, profile, size=560, fov_deg=None):
    """Top-down radar: occupancy grid, observed ground, objects, free distance per heading."""
    x_min, x_max, z_min, z_max = fs.extent
    W = size
    H = int(size * (z_max - z_min) / (x_max - x_min))
    img = np.full((H, W, 3), 32, np.uint8)
    nz, nx = fs.grid_occ.shape
    free = cv2.resize(fs.grid_free[::-1].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    occ = cv2.resize(fs.grid_occ[::-1].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    img[free] = (60, 110, 60)
    img[occ] = (210, 210, 210)

    def px(x, z):
        return int((x - x_min) / (x_max - x_min) * W), int(H - (z - z_min) / (z_max - z_min) * H)

    cam = px(0, 0)
    sc = W / 560
    # range rings
    for k in range(1, 5):
        r = z_max * k / 4
        rr = int(r / (x_max - x_min) * W)
        cv2.ellipse(img, cam, (rr, rr), 0, 180, 360, (90, 90, 90), 1, cv2.LINE_AA)
        _text(img, f"{r:.1f} m", (cam[0] + 4, cam[1] - rr + 14), 0.4 * sc, (160, 160, 160))
    # free distance per heading (polar histogram)
    pts = [cam]
    for th, d in zip(np.radians(fs.headings_deg), fs.free_dist):
        pts.append(px(d * np.sin(th), d * np.cos(th)))
    poly = np.array(pts, np.int32)
    lay = img.copy()
    cv2.fillPoly(lay, [poly], (120, 90, 40))
    img = cv2.addWeighted(lay, 0.35, img, 0.65, 0)
    cv2.polylines(img, [poly], True, (230, 160, 60), 1, cv2.LINE_AA)
    # corridor of the user / robot along the suggested heading
    th = np.radians(fs.best_heading_deg)
    d = fs.free_dist[int(np.argmin(np.abs(fs.headings_deg - fs.best_heading_deg)))]
    hw = profile.half_width_m
    c, s_ = np.cos(th), np.sin(th)
    corners = [(-hw * c, hw * s_), (hw * c, -hw * s_)]
    quad = [px(*corners[0]), px(*corners[1]),
            px(corners[1][0] + d * s_, corners[1][1] + d * c), px(corners[0][0] + d * s_, corners[0][1] + d * c)]
    cv2.polylines(img, [np.array(quad, np.int32)], True, (230, 120, 20), 2, cv2.LINE_AA)
    cv2.arrowedLine(img, cam, px(0.9 * d * s_, 0.9 * d * c), (230, 120, 20), 3, cv2.LINE_AA, tipLength=0.08)
    for o in objects:
        if not (x_min < o.x_m < x_max and 0 < o.dist_m < z_max):
            continue
        col = LEVEL_COLORS[o.level]
        p0, p1 = px(o.x_lo, o.dist_m), px(o.x_hi, o.dist_m)
        cv2.line(img, p0, p1, col, max(3, int(4 * sc)), cv2.LINE_AA)
        _text(img, f"{o.det.name} {o.dist_m:.2f}m", (p0[0], p0[1] - 8), 0.45 * sc, col, 1)
    cv2.circle(img, cam, int(7 * sc), (255, 255, 255), -1)
    _text(img, "top view", (8, 20), 0.55 * sc, (255, 255, 255))
    return img


def render_points(points, colors, yaw_deg=0.0, pitch_deg=-20.0, size=(640, 480), center=None, dist=None, f=None):
    """Tiny z-buffer splat renderer for a coloured point cloud (orbit view)."""
    W, H = size
    if center is None:
        center = np.median(points, 0)
    if dist is None:
        dist = 1.6 * np.linalg.norm(np.percentile(points, 95, 0) - np.percentile(points, 5, 0))
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    R = Rx @ Ry
    p = (points - center) @ R.T
    p[:, 2] += dist
    ok = p[:, 2] > 0.05
    p, c = p[ok], colors[ok]
    f = f or 0.9 * W
    u = (f * p[:, 0] / p[:, 2] + W / 2).astype(int)
    v = (f * p[:, 1] / p[:, 2] + H / 2).astype(int)
    keep = (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
    u, v, z, c = u[keep], v[keep], p[keep, 2], c[keep]
    order = np.argsort(-z)            # far first, near overwrite
    img = np.full((H, W, 3), 245, np.uint8)
    for du in (0, 1):
        for dv in (0, 1):
            img[v[order] + dv, u[order] + du] = c[order]
    return img


def hstack_pad(imgs, h=None, gap=6, bg=255):
    h = h or max(i.shape[0] for i in imgs)
    rs = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in imgs]
    parts = []
    for k, r in enumerate(rs):
        parts.append(r)
        if k < len(rs) - 1:
            parts.append(np.full((h, gap, 3), bg, np.uint8))
    return np.hstack(parts)


def vstack_pad(imgs, w=None, gap=6, bg=255):
    w = w or max(i.shape[1] for i in imgs)
    rs = [cv2.resize(i, (w, int(i.shape[0] * w / i.shape[1]))) for i in imgs]
    parts = []
    for k, r in enumerate(rs):
        parts.append(r)
        if k < len(rs) - 1:
            parts.append(np.full((gap, w, 3), bg, np.uint8))
    return np.vstack(parts)


def caption(img, txt, scale=None, h=None):
    w = img.shape[1]
    scale = scale or max(0.5, w / 1400)
    h = h or int(38 * scale / 0.6)
    bar = np.full((h, w, 3), 255, np.uint8)
    _text(bar, txt, (10, int(h * 0.68)), scale, (30, 30, 30), max(1, int(scale * 2)))
    return np.vstack([bar, img])
