"""Compare stereo matchers from their cached disparities (eval/cache_stereo.py).

    python eval/eval_stereo.py [--engines census ffs fs]

1. Course pairs: bad pixel ratio exactly as the course eval.py (integer disparity, error > 1,
   every pixel with ground truth, including the left border that the right camera cannot see).
2. MiddEval3 training (quarter resolution): bad-0.5 / 1 / 2 and average error on non-occluded pixels.
3. Metric depth with the true slide length on MiddEval3 + Middlebury 2021 close-range (26 scenes):
   share of pixels within 5 % / 10 % and AbsRel; per-object distance error for objects found by
   YOLO11-seg (distance = 20th percentile of depth inside the eroded mask, same rule for the truth).
Run time and GPU memory come from the cache (two passes per pair: left and right view).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_course import CONFIG, course_bad_pixel  # noqa: E402
from mbio import scenes  # noqa: E402
from mbio import read_calib, read_pfm  # noqa: E402
from stereoguard import fusion  # noqa: E402

NAMES = {"census": "Team 18 Census", "ffs": "Fast-FoundationStereo", "fs": "FoundationStereo"}


def load(engine, key):
    return np.load(os.path.join(ROOT, "outputs", "cache", f"stereo_{engine}", f"{key}.npz"))


_DET = None


def object_masks(cache_path, L):
    """YOLO11-seg masks for one scene (cached)."""
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        return list(z["names"]), z["masks"]
    from stereoguard.objects import Detector
    global _DET
    _DET = _DET or Detector("yolo11s-seg.pt", conf=0.3)
    det = _DET
    dets = det.detect(L)
    names = [d.name for d in dets]
    masks = np.array([d.mask for d in dets], bool).reshape(-1, *L.shape[:2])
    np.savez_compressed(cache_path, names=np.array(names, dtype=object), masks=masks)
    return names, masks


def course_figure(engines, res, path):
    """left | ground truth | one column per matcher, with the course bad pixel ratio."""
    rows = []
    for name, (_, sc) in CONFIG.items():
        d0 = os.path.join(ROOT, "data", "course", name)
        L = cv2.imread(os.path.join(d0, "img_left.png"))
        gt = cv2.imread(os.path.join(d0, "disp_gt.png"), cv2.IMREAD_UNCHANGED)
        col = lambda x: cv2.applyColorMap(np.uint8(np.clip(x, 0, 255)), cv2.COLORMAP_TURBO)
        g = col(gt)
        g[gt == 0] = 0
        tiles = [L, g] + [col(load(e, f"course_{name}")["dl"] * sc) for e in engines]
        tiles = [cv2.resize(t, (240, int(240 * L.shape[0] / L.shape[1]))) for t in tiles]
        caps = ["left", "ground truth"] + [f"{NAMES[e]} {100 * res['course'][e][name]:.2f}%" for e in engines]
        for t, c in zip(tiles, caps):
            cv2.rectangle(t, (0, 0), (t.shape[1], 18), (255, 255, 255), -1)
            cv2.putText(t, c, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1, cv2.LINE_AA)
        rows.append(np.hstack(tiles))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def p20(depth, mask):
    v = depth[mask & np.isfinite(depth) & (depth > 0)]
    return float(np.percentile(v, 20)) if v.size >= 30 else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engines", nargs="+", default=["census", "ffs", "fs"])
    p.add_argument("--out", default=os.path.join(ROOT, "results", "stereo"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    det_dir = os.path.join(ROOT, "outputs", "cache", "objects")
    os.makedirs(det_dir, exist_ok=True)
    res = {"course": {}, "middeval3": {}, "metric": {}, "timing": {}}

    # 1. course pairs (also without the left strip that the right camera never sees)
    res["course_visible"] = {}
    for e in a.engines:
        rows, vis = {}, {}
        for name, (_, sc) in CONFIG.items():
            gt = cv2.imread(os.path.join(ROOT, "data", "course", name, "disp_gt.png"), cv2.IMREAD_UNCHANGED)
            d = load(e, f"course_{name}")["dl"]
            d = d if e == "census" else np.clip(np.round(d), 0, 255)
            rows[name] = course_bad_pixel(d, gt, sc)
            seen = (np.arange(gt.shape[1])[None, :] - gt / sc) >= 0
            vis[name] = course_bad_pixel(np.where(seen, d, 0), np.where(seen, gt, 0), sc)
        rows["mean"] = float(np.mean(list(rows.values())))
        vis["mean"] = float(np.mean(list(vis.values())))
        res["course"][e] = rows
        res["course_visible"][e] = vis

    # 2. MiddEval3 non-occluded disparity error
    mid = os.path.join(ROOT, "data", "middeval3", "MiddEval3", "trainingQ")
    for e in a.engines:
        acc = {k: [] for k in ["bad0.5", "bad1.0", "bad2.0", "avgerr"]}
        for sc in sorted(os.listdir(mid)):
            gt = read_pfm(os.path.join(mid, sc, "disp0GT.pfm"))
            m = (cv2.imread(os.path.join(mid, sc, "mask0nocc.png"), cv2.IMREAD_GRAYSCALE) == 255) & np.isfinite(gt)
            err = np.abs(load(e, f"MiddEval3_{sc}")["dl"] - gt)[m]
            for t in (0.5, 1.0, 2.0):
                acc[f"bad{t}"].append(float(np.mean(err > t)))
            acc["avgerr"].append(float(err.mean()))
        res["middeval3"][e] = {k: float(np.mean(v)) for k, v in acc.items()}

    # 3. metric depth with the true baseline + per-object distances
    pix = {e: [] for e in a.engines}
    objs = {e: [] for e in a.engines}
    for ds, sc, lp, _, gp, cp, s in scenes(ROOT):
        cal = read_calib(cp)
        L = cv2.imread(lp)
        if s != 1.0:
            L = cv2.resize(L, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = L.shape[:2]
        f, B, doffs = cal["f"] * s, cal["baseline_m"], cal["doffs"] * s
        gd = read_pfm(gp)
        gz = cal["f"] * B / (gd + cal["doffs"])
        gz[~np.isfinite(gd) | (gd <= 0)] = np.nan
        gt = cv2.resize(gz, (w, h), interpolation=cv2.INTER_NEAREST) if s != 1.0 else gz
        gm = np.isfinite(gt)
        names, masks = object_masks(os.path.join(det_dir, f"{ds}_{sc}.npz"), L)
        kernel = np.ones((5, 5), np.uint8)
        cores = [cv2.erode(mk.astype(np.uint8), kernel).astype(bool) for mk in masks]
        cores = [c for c in cores if (c & gm).sum() >= 200]
        for e in a.engines:
            z = fusion.disparity_to_depth(load(e, f"{ds}_{sc}")["dl"], f, B, doffs=doffs)
            ok = gm & np.isfinite(z) & (z > 0)
            r = np.maximum(z[ok] / gt[ok], gt[ok] / z[ok])
            pix[e].append({"d05": float(np.mean(r < 1.05)), "d10": float(np.mean(r < 1.10)),
                           "absrel": float(np.mean(np.abs(z[ok] - gt[ok]) / gt[ok]))})
            for c in cores:
                g, pr = p20(gt, c), p20(z, c)
                if np.isfinite(g) and np.isfinite(pr):
                    objs[e].append((abs(pr - g) / g, abs(pr - g)))
    for e in a.engines:
        rel = np.array([o[0] for o in objs[e]])
        ab = np.array([o[1] for o in objs[e]])
        res["metric"][e] = {"pixels_within5": float(np.mean([r["d05"] for r in pix[e]])),
                            "pixels_within10": float(np.mean([r["d10"] for r in pix[e]])),
                            "absrel": float(np.mean([r["absrel"] for r in pix[e]])),
                            "objects": int(len(rel)), "object_mean_rel": float(rel.mean()),
                            "object_max_rel": float(rel.max()), "object_mean_abs_cm": float(100 * ab.mean())}
        tt = [float(load(e, f"{ds}_{sc}")["time"]) for ds, sc, *_ in scenes(ROOT) if ds == "MB2021"]
        pk = [float(load(e, f"{ds}_{sc}")["peak"]) for ds, sc, *_ in scenes(ROOT) if ds == "MB2021"]
        res["timing"][e] = {"pair_960x540_two_passes_s": float(np.median(tt)),
                            "peak_gpu_gb": float(np.nanmax(pk)) if np.isfinite(pk).any() else None}
    json.dump(res, open(os.path.join(a.out, "stereo_summary.json"), "w"), indent=1)
    course_figure(a.engines, res, os.path.join(ROOT, "results", "course", "course_comparison.jpg"))

    print("course bad pixel (mean):", {NAMES[e]: round(100 * v["mean"], 2) for e, v in res["course"].items()})
    print("course, visible part only:", {NAMES[e]: {k: round(100 * x, 2) for k, x in v.items()}
                                         for e, v in res["course_visible"].items()})
    for e in a.engines:
        c, m3, mt, tm = res["course"][e], res["middeval3"][e], res["metric"][e], res["timing"][e]
        print(f"{NAMES[e]:24s} course " + " ".join(f"{k} {100 * v:.2f}" for k, v in c.items())
              + f" | MiddEval3 bad1 {100 * m3['bad1.0']:.1f} bad2 {100 * m3['bad2.0']:.1f} avg {m3['avgerr']:.2f}"
              + f" | metric px<5% {100 * mt['pixels_within5']:.1f} obj {100 * mt['object_mean_rel']:.2f}% "
              + f"({mt['object_mean_abs_cm']:.2f} cm, n={mt['objects']}) | {tm['pair_960x540_two_passes_s']:.2f}s")


if __name__ == "__main__":
    main()
