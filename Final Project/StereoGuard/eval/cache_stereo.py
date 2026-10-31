"""Run one stereo matcher on every evaluation pair and cache the disparities.

    python eval/cache_stereo.py --engine fs        # FoundationStereo (ViT-L)
    python eval/cache_stereo.py --engine ffs       # Fast-FoundationStereo
    python eval/cache_stereo.py --engine census    # Team 18 matcher

Pairs: the four course pairs, MiddEval3 training (quarter resolution) and the 11 close-range
Middlebury 2021 scenes (half resolution).  Output: outputs/cache/stereo_<engine>/<set>_<scene>.npz
with the left / right disparity and the run time; GPU peak memory is stored too.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mbio import scenes  # noqa: E402
from mbio import read_calib  # noqa: E402

COURSE = {"Tsukuba": 15, "Venus": 20, "Teddy": 60, "Cones": 60}


def pairs():
    for name, md in COURSE.items():
        d = os.path.join(ROOT, "data", "course", name)
        yield "course", name, d + "/img_left.png", d + "/img_right.png", 1.0, md
    for ds, sc, lp, rp, _, cp, s in scenes(ROOT):
        yield ds, sc, lp, rp, s, int(np.ceil(read_calib(cp)["ndisp"] * s))


def make_engine(name, device):
    from stereoguard.stereo import FastFoundationStereo, FoundationStereoTorch, Team18Census
    if name == "fs":
        return FoundationStereoTorch(device=device)
    if name == "ffs":
        return FastFoundationStereo(device=device)
    return Team18Census()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, choices=["fs", "ffs", "census"])
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    out = os.path.join(ROOT, "outputs", "cache", f"stereo_{a.engine}")
    os.makedirs(out, exist_ok=True)
    eng = make_engine(a.engine, a.device)
    torch = None
    if a.engine == "fs":
        import torch
    times = {}
    for ds, sc, lp, rp, s, max_disp in pairs():
        f = os.path.join(out, f"{ds}_{sc}.npz")
        if os.path.exists(f):
            continue
        L, R = cv2.imread(lp), cv2.imread(rp)
        if s != 1.0:
            L = cv2.resize(L, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            R = cv2.resize(R, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if a.engine == "census":
            eng.max_disp = max_disp
        if a.engine != "census" and not times:
            eng(L, R)                     # warm-up: CUDA kernels / ONNX graph
        if torch is not None:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        dl, dr = eng(L, R, with_right=True)
        if torch is not None:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch is not None else float("nan")
        # for the Census matcher the second output is its own 'reliable' disparity (0 = rejected)
        key = "rel" if a.engine == "census" else "dr"
        np.savez_compressed(f, dl=dl.astype(np.float32), **{key: dr.astype(np.float32)}, time=dt, peak=peak)
        times[f"{ds}_{sc}"] = (dt, L.shape[1], L.shape[0])
        print(f"{ds:9s} {sc:12s} {L.shape[1]}x{L.shape[0]}  {dt:.2f}s (two passes)  peak {peak:.2f} GB", flush=True)
    with open(os.path.join(out, "times.json"), "a") as fo:
        fo.write(json.dumps(times) + "\n")


if __name__ == "__main__":
    main()
