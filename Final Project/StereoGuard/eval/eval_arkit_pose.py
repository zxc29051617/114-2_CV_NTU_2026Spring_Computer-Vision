"""Where do the errors on the ARKitScenes slide pairs come from?  Same pairs, true slide length, three set-ups:

    python eval/eval_arkit_pose.py

  f35        only the focal length is known (like a phone photo with EXIF); pose from the essential matrix
  K          the full intrinsic matrix is known; pose from the essential matrix
  K+pose     intrinsics and the ARKit relative pose (rectification without any estimation)
The median vertical error left after rectification shows how good each pose is.
FoundationStereo is used for all three; the metric is the share of pixels within 5 % of the laser depth.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_arkit import stats, warp_gt  # noqa: E402
from stereoguard import pipeline  # noqa: E402


def main():
    models = pipeline.Models(device="auto", stereo="fs", scale="none", focal="none", detector=None, low_vram=False)
    rows = []
    for pj in sorted(glob.glob(os.path.join(ROOT, "data", "arkitscenes", "*", "pair.json"))):
        d = os.path.dirname(pj)
        P = json.load(open(pj))
        L, R = cv2.imread(os.path.join(d, "left.png")), cv2.imread(os.path.join(d, "right.png"))
        K = np.loadtxt(os.path.join(d, "K.txt"))
        h0, w0 = L.shape[:2]
        gt0 = cv2.resize(cv2.imread(os.path.join(d, "depth_gt.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000,
                         (w0, h0), interpolation=cv2.INTER_NEAREST)
        row = {"pair": os.path.basename(d), "B": P["baseline_m"], "rot": P["rotation_deg"]}
        setups = {"f35": dict(f35=K[0, 0] * 43.267 / np.hypot(w0, h0)), "K": dict(K_full=K),
                  "K+pose": dict(K_full=K, pose=(np.array(P["R_rel"]), np.array(P["t_rel"])))}
        for name, kw in setups.items():
            try:
                res = pipeline.analyse_phone_pair(L, R, models, baseline_m=P["baseline_m"], stereo_width=w0,
                                                  profile="walking", **kw)
                gt = warp_gt(gt0, K, res.R_left, res.f, res.cx, res.cy, res.depth.shape)
                st = stats(res.depth_stereo, gt)
                row[name] = st["d05"]
                row[name + "_absrel"] = st["absrel"]
                row[name + "_dy_px"] = res.rect_stats["median_abs_dy_px"]
                row[name + "_rot_err_deg"] = None
                if name != "K+pose":
                    Rg = np.array(P["R_rel"])
                    row[name + "_rot_err_deg"] = float(np.degrees(np.linalg.norm(cv2.Rodrigues(
                        res_R(res) @ Rg.T)[0]))) if res_R(res) is not None else None
            except Exception as e:
                row[name] = None
                print(f"  {name}: {e}")
        rows.append(row)
        print(row, flush=True)
    out = {}
    for name in ["f35", "K", "K+pose"]:
        ok = [r for r in rows if r.get(name) is not None and np.isfinite(r[name])]
        v = np.array([r[name] for r in ok])
        out[name] = {"pairs": len(ok), "mean_px_within5": float(v.mean()), "pairs_over_50pct": int(np.sum(v > 0.5)),
                     "median_absrel": float(np.median([r[name + "_absrel"] for r in ok])),
                     "median_vertical_error_px": float(np.median([r[name + "_dy_px"] for r in ok]))}
    os.makedirs(os.path.join(ROOT, "results", "arkit"), exist_ok=True)
    json.dump({"summary": out, "pairs": rows}, open(os.path.join(ROOT, "results", "arkit", "pose_study.json"), "w"),
              indent=1)
    print(json.dumps(out, indent=1))


def res_R(res):
    return getattr(res, "R_rel_est", None)


if __name__ == "__main__":
    main()
