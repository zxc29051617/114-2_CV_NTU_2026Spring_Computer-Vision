"""How well MoGe-2 predicts the horizontal field of view (used when a photo has no EXIF focal length).

    python eval/eval_fov.py
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mbio import scenes  # noqa: E402
from mbio import read_calib  # noqa: E402
from stereoguard.mono import MoGe2  # noqa: E402

m = MoGe2()
errs = []
for ds, sc, lp, _, _, cp, s in scenes(ROOT):
    cal = read_calib(cp)
    img = cv2.imread(lp)
    w = img.shape[1]
    true = float(np.degrees(2 * np.arctan(w / (2 * cal["f"]))))
    _, pred = m(img, max_side=768)
    errs.append(pred - true)
    print(f"{ds:9s} {sc:12s} true {true:5.1f}  pred {pred:5.1f}  err {pred - true:+5.1f} deg", flush=True)
e = np.array(errs)
print(f"mean |err| {np.mean(np.abs(e)):.2f} deg, median |err| {np.median(np.abs(e)):.2f} deg, max {np.max(np.abs(e)):.1f} deg")
