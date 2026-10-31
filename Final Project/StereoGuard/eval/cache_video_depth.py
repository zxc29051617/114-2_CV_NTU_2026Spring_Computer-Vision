"""Metric Video Depth Anything on the KITTI left camera (single-camera video depth).

    python eval/cache_video_depth.py --encoder vitl     # or vits

Writes outputs/cache/vda_<encoder>/kitti.npz (float16 depth per frame) for eval/eval_kitti.py.
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stereoguard.mono import VideoDepthAnythingMetric  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default=os.path.join(ROOT, "data", "kitti_raw", "2011_09_26", "2011_09_26_drive_0005_sync"))
    p.add_argument("--encoder", default="vitl", choices=["vits", "vitl"])
    a = p.parse_args()
    import torch
    out = os.path.join(ROOT, "outputs", "cache", f"vda_{a.encoder}")
    os.makedirs(out, exist_ok=True)
    frames = np.stack([cv2.imread(f)[..., ::-1] for f in sorted(glob.glob(os.path.join(a.seq, "image_02", "data", "*.png")))])
    model = VideoDepthAnythingMetric(encoder=a.encoder)
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    depth = model(frames)
    dt = time.perf_counter() - t
    np.savez_compressed(os.path.join(out, "kitti.npz"), depth=depth.astype(np.float16), time=dt,
                        peak=torch.cuda.max_memory_allocated() / 1e9)
    print(f"{len(frames)} frames in {dt:.1f}s ({len(frames) / dt:.1f} fps), peak {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
