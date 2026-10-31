"""Readers for Middlebury files (PFM disparities, calib.txt)."""
import os
import re

import numpy as np


def read_pfm(path):
    with open(path, "rb") as f:
        header = f.readline().decode().strip()
        channels = 3 if header == "PF" else 1
        w, h = map(int, f.readline().decode().split())
        scale = float(f.readline().decode().strip())
        endian = "<" if scale < 0 else ">"
        data = np.frombuffer(f.read(), dtype=endian + "f4", count=w * h * channels)
    shape = (h, w, 3) if channels == 3 else (h, w)
    return np.flipud(data.reshape(shape)).astype(np.float32)


def read_calib(path):
    txt = open(path).read()
    cam0 = re.search(r"cam0=\[([^\]]+)\]", txt).group(1).replace(";", " ").split()
    vals = dict(re.findall(r"^(\w+)=([^\n\[]+)$", txt, re.M))
    K = np.array([float(v) for v in cam0]).reshape(3, 3)
    return {
        "f": K[0, 0], "cx": K[0, 2], "cy": K[1, 2],
        "baseline_m": float(vals["baseline"]) / 1000.0,
        "doffs": float(vals.get("doffs", 0.0)),
        "ndisp": int(float(vals.get("ndisp", 256))),
        "vmin": float(vals["vmin"]) if "vmin" in vals else None,
        "vmax": float(vals["vmax"]) if "vmax" in vals else None,
        "width": int(vals["width"]), "height": int(vals["height"]),
    }


# The 11 close-range Middlebury 2021 scenes.  The 13 room-scale scenes are left out: their calib.txt
# baselines make everyday objects several times too large (artroom1's door would be ~6 m high).
MB21_CLOSE = ["chess1", "chess2", "chess3", "curule1", "curule2", "curule3", "skates1", "skates2",
              "skiboots1", "skiboots2", "skiboots3"]


def scenes(root):
    """(dataset, scene, left, right, gt disparity, calib, resize factor) for the metric evaluations."""
    mid = os.path.join(root, "data", "middeval3", "MiddEval3", "trainingQ")
    out = []
    for s in sorted(os.listdir(mid)):
        d = os.path.join(mid, s)
        out.append(("MiddEval3", s, d + "/im0.png", d + "/im1.png", d + "/disp0GT.pfm", d + "/calib.txt", 1.0))
    for s in MB21_CLOSE:
        d = os.path.join(root, "data", "mb2021", s)
        out.append(("MB2021", s, d + "/im0.png", d + "/im1.png", d + "/disp0.pfm", d + "/calib.txt", 0.5))
    return out


