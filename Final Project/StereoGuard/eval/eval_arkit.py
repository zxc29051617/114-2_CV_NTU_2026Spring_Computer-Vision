"""Phone-camera check on ARKitScenes slide pairs (scripts/download_arkitscenes.py).

    python eval/eval_arkit.py

Every pair is two frames of an iPad video whose ARKit poses say the camera slid sideways, with a
laser-scanner depth map for the left frame.  StereoGuard runs exactly as on a phone pair (SIFT +
essential matrix rectification with the frame's intrinsics as 'EXIF', FoundationStereo, YOLO11-seg):
  known  slide length from the ARKit poses
  est.   slide length estimated by the Depth Pro / UniDepthV2 / DA-V2-L ensemble; DA-V2-S, MoGe-2 S/L and
         Metric3D v2 are added afterwards for eval/eval_scale_fusion.py
The ground truth is warped into the rectified left view.  Reported: share of pixels within 5 % / 10 %,
AbsRel, and per-object distance error (20th percentile of depth inside the mask).
Intermediate data for the scale study is cached in outputs/cache/arkit/.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stereoguard import fusion, pipeline  # noqa: E402


def warp_gt(gt, K, R1, f, cx, cy, shape):
    """Laser depth of the original left frame -> depth seen by the rectified left camera."""
    h0, w0 = gt.shape
    v, u = np.nonzero(gt > 0)
    z = gt[v, u]
    X = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1) @ R1.T
    ok = X[:, 2] > 0.05
    X = X[ok]
    uu = np.round(f * X[:, 0] / X[:, 2] + cx).astype(int)
    vv = np.round(f * X[:, 1] / X[:, 2] + cy).astype(int)
    h, w = shape
    m = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    out = np.full(shape, np.inf, np.float32)
    np.minimum.at(out, (vv[m], uu[m]), X[m, 2].astype(np.float32))
    out[~np.isfinite(out)] = np.nan
    return out


def p20(d, m):
    v = d[m & np.isfinite(d) & (d > 0)]
    return float(np.percentile(v, 20)) if v.size >= 30 else float("nan")


def stats(pred, gt):
    ok = np.isfinite(gt) & np.isfinite(pred) & (pred > 0)
    r = np.maximum(pred[ok] / gt[ok], gt[ok] / pred[ok])
    return {"d05": float(np.mean(r < 1.05)), "d10": float(np.mean(r < 1.10)),
            "absrel": float(np.mean(np.abs(pred[ok] - gt[ok]) / gt[ok])), "n": int(ok.sum())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "arkitscenes"))
    p.add_argument("--out", default=os.path.join(ROOT, "results", "arkit"))
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(ROOT, "outputs", "cache", "arkit")
    os.makedirs(cache, exist_ok=True)
    models = pipeline.Models(device=a.device, focal="none", scale="depthpro+unidepth+dav2l", low_vram=False)
    rows = []
    for d in sorted(glob.glob(os.path.join(a.data, "*", "pair.json"))):
        d = os.path.dirname(d)
        name = os.path.basename(d)
        pj = json.load(open(os.path.join(d, "pair.json")))
        L, R = cv2.imread(os.path.join(d, "left.png")), cv2.imread(os.path.join(d, "right.png"))
        K = np.loadtxt(os.path.join(d, "K.txt"))
        h0, w0 = L.shape[:2]
        gt0 = cv2.imread(os.path.join(d, "depth_gt.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        gt0 = cv2.resize(gt0, (w0, h0), interpolation=cv2.INTER_NEAREST)
        f35 = K[0, 0] * 43.267 / np.hypot(w0, h0)        # the frame's focal length plays the role of EXIF
        try:
            res = pipeline.analyse_phone_pair(L, R, models, baseline_m=None, f35=f35, profile="walking",
                                              stereo_width=w0)
        except Exception as e:
            print(f"{name}: failed ({e})")
            continue
        if getattr(res, "swapped", False):
            print(f"{name}: views swapped, skipped")
            continue
        gt = warp_gt(gt0, K, res.R_left, res.f, res.cx, res.cy, res.depth.shape)
        votes = dict(res.scale_votes)
        ens = float(np.exp(np.mean(np.log([votes[k] for k in ("depthpro", "unidepth", "dav2l")]))))
        B = pj["baseline_m"]
        depth_known = res.depth * (B / res.baseline_m)
        depth_est = res.depth * (ens / res.baseline_m)
        row = {"pair": name, "video": pj["video"], "B": B, "B_votes": votes, "B_ensemble": ens,
               "rot_deg": pj["rotation_deg"], "gt_median_m": float(np.nanmedian(gt)),
               "rect_dy_px": res.rect_stats["median_abs_dy_px"],
               "known": stats(depth_known, gt), "est": stats(depth_est, gt), "objects": []}
        kernel = np.ones((5, 5), np.uint8)
        for o in res.objects:
            core = cv2.erode(o.det.mask.astype(np.uint8), kernel).astype(bool)
            g = p20(gt, core)
            if not np.isfinite(g) or (core & np.isfinite(gt)).sum() < 200:
                continue
            row["objects"].append({"name": o.det.name, "gt": g, "known": p20(depth_known, core),
                                   "est": p20(depth_est, core)})
        np.savez_compressed(os.path.join(cache, f"{name}.npz"), image=res.image, disp=res.disp,
                            confident=res.confident, gt=gt, f=res.f, cx=res.cx, cy=res.cy, B=B,
                            masks=np.array([o.det.mask for o in res.objects], bool).reshape(-1, *res.depth.shape),
                            names=np.array([o.det.name for o in res.objects], dtype=object))
        rows.append(row)
        print(f"{name}: B {100 * B:.1f} cm, est {100 * ens:.1f} cm, GT median {row['gt_median_m']:.2f} m, "
              f"known px<5% {100 * row['known']['d05']:.0f}%, est px<5% {100 * row['est']['d05']:.0f}%, "
              f"{len(row['objects'])} objects", flush=True)
    # the other single-image models, one at a time over the cached rectified views
    models.release("fs")
    for m in ("depthpro", "unidepth", "dav2l"):
        models.release(m)
    for extra in ("dav2s", "moge2", "moge2l", "metric3d"):
        net = models.get(extra)
        for r in rows:
            z = np.load(os.path.join(cache, f"{r['pair']}.npz"), allow_pickle=True)
            img, f = z["image"], float(z["f"])
            hfov = float(np.degrees(2 * np.arctan(img.shape[1] / (2 * f))))
            zm, _ = net(img, hfov_deg=hfov)
            b, _ = fusion.estimate_baseline(z["disp"], f, zm, z["confident"])
            r["B_votes"][extra] = b
        models.release(extra)
        print(f"{extra}: done", flush=True)
    json.dump(rows, open(os.path.join(a.out, "arkit_pairs.json"), "w"), indent=1, default=float)

    def agg(mode):
        px = [r[mode] for r in rows]
        ob = np.array([abs(o[mode] - o["gt"]) / o["gt"] for r in rows for o in r["objects"] if np.isfinite(o[mode])])
        ab = np.array([abs(o[mode] - o["gt"]) for r in rows for o in r["objects"] if np.isfinite(o[mode])])
        return {"pixels_within5": float(np.mean([x["d05"] for x in px])),
                "pixels_within10": float(np.mean([x["d10"] for x in px])),
                "absrel": float(np.mean([x["absrel"] for x in px])),
                "objects": int(ob.size), "object_mean_rel": float(ob.mean()) if ob.size else None,
                "object_median_rel": float(np.median(ob)) if ob.size else None,
                "object_within10": float(np.mean(ob < 0.1)) if ob.size else None,
                "object_mean_abs_m": float(ab.mean()) if ab.size else None}
    summary = {"pairs": len(rows), "videos": len({r["video"] for r in rows}),
               "slide_cm": [float(100 * np.min([r["B"] for r in rows])), float(100 * np.max([r["B"] for r in rows]))],
               "known_baseline": agg("known"), "estimated_baseline": agg("est"),
               "ensemble_baseline_error": {
                   "mean": float(np.mean([abs(r["B_ensemble"] / r["B"] - 1) for r in rows])),
                   "median": float(np.median([abs(r["B_ensemble"] / r["B"] - 1) for r in rows]))}}
    json.dump(summary, open(os.path.join(a.out, "arkit_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
