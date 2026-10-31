"""Where can the metric scale (length of the phone slide) and the focal length come from?

    python eval/eval_scale.py --models dav2s dav2l moge2s moge2l depthpro unidepth mapanything vggt

For every metric scene (mbio.scenes: MiddEval3 + Middlebury 2021 close-range, 26 scenes):
  * slide length  B_hat = median( Z_model * (d + doffs) / f )  over pixels that pass the left-right
    check, with d the FoundationStereo disparity (outputs/cache/stereo_fs) and Z_model the metric depth
    of a single-image model (true field of view given where the model accepts it).  For MapAnything the
    distance between its two predicted camera centres is used as well.  With an accurate disparity the
    distance error of every object equals the error of B_hat.
  * focal length predicted from the image alone (models that estimate the camera).
Results: results/scale/scale_summary.json and per-scene CSV.
"""
import argparse
import gc
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
from stereoguard import fusion  # noqa: E402

LABELS = {"dav2s": "Depth Anything V2 Metric-S", "dav2l": "Depth Anything V2 Metric-L", "moge2s": "MoGe-2 ViT-S",
          "moge2l": "MoGe-2 ViT-L", "depthpro": "Depth Pro", "unidepth": "UniDepthV2-L",
          "mapanything": "MapAnything (2 views)", "vggt": "VGGT (2 views)", "metric3d": "Metric3D v2 ViT-L"}


def build(name, device):
    from stereoguard import mono, multiview
    if name == "dav2s":
        return mono.DepthAnythingV2Metric(size="Small", device=device)
    if name == "dav2l":
        return mono.DepthAnythingV2Metric(size="Large", device=device)
    if name == "moge2s":
        return mono.MoGe2(device=device)
    if name == "moge2l":
        return mono.MoGe2(repo="Ruicheng/moge-2-vitl-normal", device=device)
    if name == "depthpro":
        return mono.DepthPro(device=device)
    if name == "unidepth":
        return mono.UniDepthV2(device=device)
    if name == "metric3d":
        return mono.Metric3Dv2(device=device)
    if name == "mapanything":
        return multiview.MapAnythingModel(device=device)
    if name == "vggt":
        return multiview.VGGTModel(device=device)
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(LABELS))
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "scale"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache_dir = os.path.join(ROOT, "outputs", "cache", "scale")
    os.makedirs(cache_dir, exist_ok=True)
    import torch

    data = []
    for ds, sc, lp, rp, _, cp, s in scenes(ROOT):
        cal = read_calib(cp)
        L, R = cv2.imread(lp), cv2.imread(rp)
        if s != 1.0:
            L = cv2.resize(L, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            R = cv2.resize(R, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        z = np.load(os.path.join(ROOT, "outputs", "cache", "stereo_fs", f"{ds}_{sc}.npz"))
        conf = fusion.lr_consistency(z["dl"], z["dr"]) & (z["dl"] > 0.5)
        data.append(dict(ds=ds, sc=sc, L=L, R=R, f=cal["f"] * s, B=cal["baseline_m"], doffs=cal["doffs"] * s,
                         d=z["dl"], conf=conf,
                         hfov=float(np.degrees(2 * np.arctan(L.shape[1] / (2 * cal["f"] * s))))))

    for name in a.models:
        cache = os.path.join(cache_dir, f"{name}.json")
        if os.path.exists(cache):
            continue
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        model = build(name, a.device)
        load_s = time.perf_counter() - t0
        rows, times = [], []
        for item in data:
            row = {"dataset": item["ds"], "scene": item["sc"], "B": item["B"], "f": item["f"]}
            t = time.perf_counter()
            if name in ("mapanything", "vggt"):
                res = model([item["L"], item["R"]])
                row["f_pred"] = float(res.K[0][0, 0])
                if res.metric:
                    row["B_pose"] = res.baseline()
                    zm = res.depth[0]
                    row["B_depth"], _ = fusion.estimate_baseline(item["d"], item["f"], zm, item["conf"],
                                                                 doffs=item["doffs"])
            else:
                zm, _ = model(item["L"], hfov_deg=item["hfov"])            # true FoV where supported
                row["B_depth"], _ = fusion.estimate_baseline(item["d"], item["f"], zm, item["conf"],
                                                             doffs=item["doffs"])
                if name not in ("dav2s", "dav2l", "metric3d"):
                    _, fov = model(item["L"])                                # camera from the image only
                    row["f_pred"] = item["L"].shape[1] / (2 * np.tan(np.radians(fov) / 2))
            times.append(time.perf_counter() - t)
            rows.append(row)
            print(f"{name:12s} {item['ds']:9s} {item['sc']:12s} B {100 * item['B']:6.2f} cm  "
                  + "  ".join(f"{k} {100 * v:6.2f}" for k, v in row.items() if k.startswith("B_") and v)
                  + (f"  f {row['f_pred']:.0f}/{item['f']:.0f}" if "f_pred" in row else ""), flush=True)
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")
        json.dump({"rows": rows, "time_s": float(np.median(times)), "load_s": load_s, "peak_gb": peak},
                  open(cache, "w"), indent=1)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- summary ----
    summary = {}
    for name in a.models:
        c = json.load(open(os.path.join(cache_dir, f"{name}.json")))
        rows = c["rows"]
        entry = {"label": LABELS[name], "time_s": c["time_s"], "peak_gb": c["peak_gb"]}
        for key in ("B_depth", "B_pose"):
            vals = [(r[key], r["B"], r["dataset"]) for r in rows if r.get(key)]
            if not vals:
                continue
            e = np.array([abs(b / g - 1) for b, g, _ in vals])
            entry[key] = {"mean_abs_rel": float(e.mean()), "median_abs_rel": float(np.median(e)),
                          "within10": float(np.mean(e < 0.1)), "within20": float(np.mean(e < 0.2))}
            for ds in ("MiddEval3", "MB2021"):
                ee = np.array([abs(b / g - 1) for b, g, d in vals if d == ds])
                entry[key][ds] = float(ee.mean())
        fr = [(r["f_pred"], r["f"], r["dataset"]) for r in rows if "f_pred" in r]
        if fr:
            e = np.array([abs(fp / fg - 1) for fp, fg, _ in fr])
            entry["focal"] = {"mean_abs_rel": float(e.mean()), "median_abs_rel": float(np.median(e)),
                              "MB2021": float(np.mean([abs(fp / fg - 1) for fp, fg, d in fr if d == "MB2021"]))}
        summary[name] = entry
    json.dump(summary, open(os.path.join(a.out, "scale_summary.json"), "w"), indent=1)
    for name, e in summary.items():
        line = f"{e['label']:30s}"
        if "B_depth" in e:
            line += f" B(depth) mean {100 * e['B_depth']['mean_abs_rel']:5.1f}% med {100 * e['B_depth']['median_abs_rel']:5.1f}%"
        if "B_pose" in e:
            line += f" B(pose) mean {100 * e['B_pose']['mean_abs_rel']:5.1f}%"
        if "focal" in e:
            line += f" focal mean {100 * e['focal']['mean_abs_rel']:5.1f}% (phone {100 * e['focal']['MB2021']:4.1f}%)"
        line += f"  {e['time_s']:.2f}s  {e['peak_gb']:.1f} GB"
        print(line)


if __name__ == "__main__":
    main()
