"""Course test pairs: bad pixel ratio and runtime of the Team 18 matcher vs Fast-FoundationStereo.

    python eval/eval_course.py [--device cpu|cuda] [--threads 4]

Bad pixel ratio is computed exactly like the course eval.py (integer disparity, error > 1, pixels
with ground truth only).  For the deep model the float disparity is rounded to the nearest
integer first; a sub-pixel version (|d - gt| > 1 with the float output) is also reported.
"""
import argparse
import importlib.util
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stereoguard.stereo import FastFoundationStereo  # noqa: E402

TEAM18_CORE = os.path.join(os.path.dirname(ROOT), "Stereo Matching Team_Team18", "Stereo Matching")
CONFIG = {"Tsukuba": (15, 16), "Venus": (20, 8), "Teddy": (60, 4), "Cones": (60, 4)}


def load(path, name):
    sys.dont_write_bytecode = True          # do not leave __pycache__ in the original folder
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def course_bad_pixel(disp, gt_raw, scale, threshold=1.0):
    """Vectorised copy of the course eval.evaluate()."""
    d = np.int32(np.uint8(np.clip(disp * scale, 0, 255)) / scale)
    g = np.int32(gt_raw / scale)
    m = g > 0
    return float(np.mean(np.abs(g[m] - d[m]) > threshold))


def subpixel_bad(disp, gt_raw, scale, threshold=1.0):
    g = gt_raw.astype(np.float32) / scale
    m = g > 0
    return float(np.mean(np.abs(g[m] - disp[m]) > threshold))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(ROOT, "data", "course"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out", default=os.path.join(ROOT, "results", "course"))
    p.add_argument("--repeat", type=int, default=3)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    team18 = load(os.path.join(TEAM18_CORE, "computeDisp.py"), "team18_core")
    ffs = FastFoundationStereo(device=a.device, threads=a.threads)

    rows = []
    vis = []
    for name, (max_disp, scale) in CONFIG.items():
        d = os.path.join(a.data, name)
        L = cv2.imread(os.path.join(d, "img_left.png"))
        R = cv2.imread(os.path.join(d, "img_right.png"))
        gt = cv2.imread(os.path.join(d, "disp_gt.png"), cv2.IMREAD_UNCHANGED)
        if gt.ndim == 3:
            gt = gt[..., 0]

        t_c = []
        for _ in range(a.repeat):
            t = time.perf_counter()
            dc = team18.computeDisp(L, R, max_disp)
            t_c.append(time.perf_counter() - t)
        ffs(L, R)  # warm-up
        t_d = []
        for _ in range(a.repeat):
            t = time.perf_counter()
            dd, _ = ffs(L, R)
            t_d.append(time.perf_counter() - t)
        dd_int = np.clip(np.round(dd), 0, 255)

        row = {
            "pair": name, "size": f"{L.shape[1]}x{L.shape[0]}",
            "census_bad": course_bad_pixel(dc.astype(np.float32), gt, scale),
            "ffs_bad": course_bad_pixel(dd_int, gt, scale),
            "census_bad_subpix": subpixel_bad(dc.astype(np.float32), gt, scale),
            "ffs_bad_subpix": subpixel_bad(dd, gt, scale),
            "census_time": float(np.median(t_c)), "ffs_time": float(np.median(t_d)),
        }
        rows.append(row)
        print(f"{name:8s} census {100 * row['census_bad']:6.2f}%  ffs {100 * row['ffs_bad']:6.2f}%  "
              f"time {row['census_time']:.2f}s / {row['ffs_time']:.2f}s")
        cv2.imwrite(os.path.join(a.out, f"{name}_census.png"), np.uint8(np.clip(dc * scale, 0, 255)))
        cv2.imwrite(os.path.join(a.out, f"{name}_ffs.png"), np.uint8(np.clip(dd_int * scale, 0, 255)))
        vis.append((name, L, gt, dc * scale, dd_int * scale, row))

    with open(os.path.join(a.out, "course_results.csv"), "w") as f:
        keys = list(rows[0])
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(round(r[k], 5)) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    avg_c = np.mean([r["census_bad"] for r in rows])
    avg_d = np.mean([r["ffs_bad"] for r in rows])
    print(f"average  census {100 * avg_c:.2f}%  ffs {100 * avg_d:.2f}%")

    # comparison figure: left | GT | Team 18 | Fast-FoundationStereo
    tiles = []
    for name, L, gt, dc, dd, row in vis:
        g = lambda x: cv2.applyColorMap(np.uint8(np.clip(x, 0, 255)), cv2.COLORMAP_TURBO)
        gt_c = g(gt)
        gt_c[gt == 0] = 0
        r = [L, gt_c, g(dc), g(dd)]
        tiles.append(np.hstack([cv2.resize(x, (300, int(300 * L.shape[0] / L.shape[1]))) for x in r]))
        lab = np.full((28, tiles[-1].shape[1], 3), 255, np.uint8)
        txt = (f"{name}: left | ground truth | Team 18 Census {100 * row['census_bad']:.2f}% | "
               f"Fast-FoundationStereo {100 * row['ffs_bad']:.2f}%")
        cv2.putText(lab, txt, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        tiles[-1] = np.vstack([lab, tiles[-1]])
    cv2.imwrite(os.path.join(a.out, "course_comparison.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])


if __name__ == "__main__":
    main()
