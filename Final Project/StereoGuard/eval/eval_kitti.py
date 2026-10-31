"""Outdoor distance, closing speed and time-to-collision on a KITTI raw sequence (LiDAR as reference).

    python eval/eval_kitti.py [--seq data/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync] [--device auto]

Every frame: YOLO11-seg + ByteTrack tracks objects; the Velodyne points that fall inside the eroded
mask give the reference distance (20th percentile, at least 8 points).  The same statistic is taken
from (a) Fast-FoundationStereo depth with the KITTI rig (0.54 m baseline) and (b) MoGe-2 on the left
image only (what a single phone camera would give), sampled at the same LiDAR pixels.
Closing speed and TTC use the same 1 s least-squares window for every source; the scale-expansion
TTC (Lee's tau, no depth at all) is also reported.
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
from run_video import kitti_calib  # noqa: E402
from stereoguard import fusion  # noqa: E402
from stereoguard.video import TrackHistory  # noqa: E402


def read_calib_file(path):
    out = {}
    for line in open(path):
        k, _, v = line.partition(":")
        try:
            out[k] = np.array(v.split(), float)
        except ValueError:
            pass
    return out


def velo_to_image(seq):
    date_dir = os.path.dirname(os.path.normpath(seq))
    c2c = read_calib_file(os.path.join(date_dir, "calib_cam_to_cam.txt"))
    v2c = read_calib_file(os.path.join(date_dir, "calib_velo_to_cam.txt"))
    T = np.eye(4)
    T[:3, :3] = v2c["R"].reshape(3, 3)
    T[:3, 3] = v2c["T"]
    Rr = np.eye(4)
    Rr[:3, :3] = c2c["R_rect_00"].reshape(3, 3)
    return c2c["P_rect_02"].reshape(3, 4) @ Rr @ T


def lidar_depth(bin_path, M, shape):
    pts = np.fromfile(bin_path, np.float32).reshape(-1, 4)[:, :3]
    pts = pts[pts[:, 0] > 1.0]
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ M.T
    z = p[:, 2]
    u = np.round(p[:, 0] / z).astype(int)
    v = np.round(p[:, 1] / z).astype(int)
    h, w = shape
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (z > 0)
    D = np.full(shape, np.nan, np.float32)
    order = np.argsort(-z[ok])          # nearest point wins
    D[v[ok][order], u[ok][order]] = z[ok][order]
    return D


def timestamps(seq):
    out = []
    for line in open(os.path.join(seq, "image_02", "timestamps.txt")):
        hh, mm, ss = line.split()[1].split(":")
        out.append(3600 * int(hh) + 60 * int(mm) + float(ss))
    return np.array(out) - out[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq", default=os.path.join(ROOT, "data", "kitti_raw", "2011_09_26", "2011_09_26_drive_0005_sync"))
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "kitti"))
    p.add_argument("--cache", default=os.path.join(ROOT, "outputs", "eval_kitti"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    from stereoguard.mono import MoGe2
    from stereoguard.objects import Detector
    from stereoguard.stereo import FastFoundationStereo, resolve_device
    ffs = FastFoundationStereo(device=resolve_device(a.device))
    moge = MoGe2()
    det = Detector("yolo11s-seg.pt", conf=0.35, imgsz=1280)
    f, cx, cy, B = kitti_calib(a.seq)
    M = velo_to_image(a.seq)
    ls = sorted(glob.glob(os.path.join(a.seq, "image_02", "data", "*.png")))
    rs = sorted(glob.glob(os.path.join(a.seq, "image_03", "data", "*.png")))
    vs = sorted(glob.glob(os.path.join(a.seq, "velodyne_points", "data", "*.bin")))
    ts = timestamps(a.seq)
    kernel = np.ones((5, 5), np.uint8)
    # optional single-camera video depth (eval/cache_video_depth.py)
    video_depth = {}
    for enc in ("vits", "vitl"):
        f_vda = os.path.join(ROOT, "outputs", "cache", f"vda_{enc}", "kitti.npz")
        if os.path.exists(f_vda):
            video_depth[f"vda_{enc}"] = np.load(f_vda)["depth"]
    rows = []
    for i in range(len(ls)):
        L, R = cv2.imread(ls[i]), cv2.imread(rs[i])
        h, w = L.shape[:2]
        cache = os.path.join(a.cache, f"{i:04d}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            z_st, z_mono = z["st"], z["mono"]
        else:
            disp, _ = ffs(L, R)
            z_st = fusion.disparity_to_depth(disp, f, B)
            z_mono, _ = moge(L, hfov_deg=float(np.degrees(2 * np.arctan(w / (2 * f)))))
            np.savez_compressed(cache, st=z_st.astype(np.float16), mono=z_mono.astype(np.float16))
            z_st, z_mono = z_st.astype(np.float16), z_mono.astype(np.float16)
        z_st, z_mono = z_st.astype(np.float32), z_mono.astype(np.float32)
        D = lidar_depth(vs[i], M, (h, w))
        pix = np.isfinite(D)
        for o in det.track(L):
            core = cv2.erode(o.mask.astype(np.uint8), kernel).astype(bool)
            sel = core & pix
            if sel.sum() < 8 or o.track_id < 0:
                continue
            row = {"frame": i, "t": float(ts[i]), "id": int(o.track_id), "name": o.name, "n_lidar": int(sel.sum()),
                   "size": float(np.sqrt(o.mask.sum())),
                   "lidar": float(np.percentile(D[sel], 20)),
                   "stereo": float(np.nanpercentile(z_st[sel], 20)),
                   "mono": float(np.nanpercentile(z_mono[sel], 20)),
                   "stereo_app": float(np.nanpercentile(z_st[core], 20))}
            for k, vd in video_depth.items():
                row[k] = float(np.nanpercentile(vd[i].astype(np.float32)[sel], 20))
            rows.append(row)
        if i % 10 == 0:
            print(f"frame {i}: {len(rows)} measurements", flush=True)

    # ---- distance accuracy ----
    gt = np.array([r["lidar"] for r in rows])
    summary = {"n_measurements": len(rows), "n_tracks": len({r["id"] for r in rows}), "baseline_m": B,
               "distance": {}, "ttc": {}}
    bins = [(0, 10), (10, 20), (20, 80)]
    sources = ["stereo", "stereo_app", "mono"] + sorted(video_depth)
    for m in sources:
        pred = np.array([r[m] for r in rows])
        e = np.abs(pred - gt) / gt
        summary["distance"][m] = {
            "mean_rel": float(np.nanmean(e)), "median_rel": float(np.nanmedian(e)),
            "within10": float(np.nanmean(e < 0.10)), "mean_abs_m": float(np.nanmean(np.abs(pred - gt))),
            "by_range": {f"{lo}-{hi}m": (float(np.nanmean(e[(gt >= lo) & (gt < hi)])) if ((gt >= lo) & (gt < hi)).any()
                                        else None) for lo, hi in bins},
            "n_by_range": {f"{lo}-{hi}m": int(((gt >= lo) & (gt < hi)).sum()) for lo, hi in bins}}

    # ---- closing speed / TTC with the same 1 s window for every source ----
    ttc_sources = ["stereo", "mono"] + sorted(video_depth)
    hists = {m: TrackHistory(1.0) for m in ["lidar"] + ttc_sources}
    pairs = []
    for r in sorted(rows, key=lambda r: r["frame"]):
        k = {}
        for m, hst in hists.items():
            hst.update(r["id"], r["t"], r[m], 0.0, r["size"])
            k[m] = hst.kinematics(r["id"])
        if any(v is None for v in k.values()):
            continue
        v_gt = -k["lidar"]["vz"]
        if v_gt < 1.0:                      # only objects that are clearly getting closer
            continue
        ttc_gt = k["lidar"]["z"] / v_gt
        if ttc_gt > 15:
            continue
        rec = {"frame": r["frame"], "id": r["id"], "name": r["name"], "ttc_gt": ttc_gt, "v_gt": v_gt}
        for m in ttc_sources:
            v = -k[m]["vz"]
            rec[f"v_{m}"] = v
            rec[f"ttc_{m}"] = k[m]["z"] / v if v > 0.1 else float("inf")
        rec["ttc_scale"] = k["stereo"]["ttc_scale"]
        pairs.append(rec)
    for m in ttc_sources + ["scale"]:
        tt = np.array([p_[f"ttc_{m}"] for p_ in pairs])
        tg = np.array([p_["ttc_gt"] for p_ in pairs])
        ok = np.isfinite(tt)
        e = np.abs(tt[ok] - tg[ok]) / tg[ok]
        summary["ttc"][m] = {"n": int(len(tg)), "finite": float(ok.mean()) if len(tg) else 0.0,
                             "median_rel": float(np.median(e)) if e.size else None,
                             "within20": float(np.mean(e < 0.2)) if e.size else None}
    for m in ttc_sources:
        vv = np.array([p_[f"v_{m}"] for p_ in pairs])
        vg = np.array([p_["v_gt"] for p_ in pairs])
        summary["ttc"][f"closing_speed_{m}_mae_mps"] = float(np.mean(np.abs(vv - vg))) if len(vg) else None
    with open(os.path.join(a.out, "kitti_eval.json"), "w") as fo:
        json.dump({"summary": summary}, fo, indent=1)
    with open(os.path.join(a.out, "kitti_measurements.csv"), "w") as fo:
        keys = list(rows[0])
        fo.write(",".join(keys) + "\n")
        for r in rows:
            fo.write(",".join(str(round(r[k], 3)) if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
