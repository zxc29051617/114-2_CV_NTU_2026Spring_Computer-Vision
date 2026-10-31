"""Video mode: tracking, distance, closing speed and time-to-collision on every frame.

Stereo sequence in KITTI raw format (left = image_02, right = image_03):
    python run_video.py --kitti data/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync --profile driving
Single phone video (metric depth from one camera):
    python run_video.py --video walk.mp4 --profile walking --count 300
"""
import argparse
import glob
import json
import os
import time

import cv2
import numpy as np

from stereoguard import freespace, pipeline, viz, video


def kitti_calib(seq_dir):
    """Focal length, principal point and baseline of the rectified color cameras 02/03."""
    date_dir = os.path.dirname(os.path.normpath(seq_dir))
    P = {}
    for line in open(os.path.join(date_dir, "calib_cam_to_cam.txt")):
        k, _, v = line.partition(":")
        if k in ("P_rect_02", "P_rect_03"):
            P[k] = np.array(v.split(), float).reshape(3, 4)
    f, cx, cy = P["P_rect_02"][0, 0], P["P_rect_02"][0, 2], P["P_rect_02"][1, 2]
    baseline = (P["P_rect_02"][0, 3] - P["P_rect_03"][0, 3]) / f
    return f, cx, cy, baseline


def frames_kitti(seq_dir, start=0, count=None, step=1):
    ls = sorted(glob.glob(os.path.join(seq_dir, "image_02", "data", "*.png")))
    rs = sorted(glob.glob(os.path.join(seq_dir, "image_03", "data", "*.png")))
    ts = [l.strip() for l in open(os.path.join(seq_dir, "image_02", "timestamps.txt"))]
    sel = range(start, len(ls) if count is None else min(len(ls), start + count), step)
    t0 = None
    for i in sel:
        hh, mm, ss = ts[i].split()[1].split(":")
        t = 3600 * int(hh) + 60 * int(mm) + float(ss)
        t0 = t if t0 is None else t0
        yield i, t - t0, cv2.imread(ls[i]), cv2.imread(rs[i])


def frames_video(path, step=1, max_width=960, count=None):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i = n = 0
    while True:
        ok, fr = cap.read()
        if not ok or (count and n >= count):
            break
        if i % step == 0:
            if fr.shape[1] > max_width:
                fr = cv2.resize(fr, (max_width, int(fr.shape[0] * max_width / fr.shape[1])), interpolation=cv2.INTER_AREA)
            yield i, i / fps, fr, None
            n += 1
        i += 1


def compose(ar, bev, width=None):
    """AR frame with the top-view radar as an inset in the lower-right corner."""
    out = ar.copy()
    h, w = out.shape[:2]
    bh = int(h * 0.55)
    bw = int(bev.shape[1] * bh / bev.shape[0])
    if bw > w * 0.35:
        bw = int(w * 0.35)
        bh = int(bev.shape[0] * bw / bev.shape[1])
    small = cv2.resize(bev, (bw, bh), interpolation=cv2.INTER_AREA)
    out[h - bh - 8:h - 8, w - bw - 8:w - 8] = small
    cv2.rectangle(out, (w - bw - 9, h - bh - 9), (w - 8, h - 8), (255, 255, 255), 1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--kitti", help="KITTI raw *_sync folder")
    src.add_argument("--video", help="monocular video file")
    p.add_argument("--out", default="outputs/video")
    p.add_argument("--profile", default="driving", choices=list(freespace.PROFILES))
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=None)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--scale", type=float, default=1.0, help="resize factor for the stereo frames")
    p.add_argument("--device", default="auto", help="auto: GPU if >= 5 GB of it is free")
    p.add_argument("--stereo", default="ffs", choices=["ffs", "fs"], help="stereo video: fast model by default")
    p.add_argument("--mono", default="moge2", choices=["moge2", "vda"],
                   help="single-camera depth: MoGe-2 per frame or metric Video Depth Anything over the clip")
    p.add_argument("--focal", default="auto", help="field of view of a phone video: vggt, moge2, ...")
    p.add_argument("--detector", default="yolo11s-seg.pt")
    p.add_argument("--gif_width", type=int, default=640)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    prof = freespace.PROFILES[a.profile]
    models = pipeline.Models(device=a.device, stereo=a.stereo, detector=a.detector, scale="none", focal=a.focal)
    hist = video.TrackHistory(window_s=1.0)
    video_depth = None
    if a.kitti:
        f, cx, cy, B = kitti_calib(a.kitti)
        f, cx, cy = f * a.scale, (cx + 0.5) * a.scale - 0.5, (cy + 0.5) * a.scale - 0.5
        frames = frames_kitti(a.kitti, a.start, a.count, a.step)
    else:
        # single camera: field of view once from the first frames, then metric depth per frame
        # (MoGe-2 with that field of view; it beat Video Depth Anything on KITTI, see README)
        frames = list(frames_video(a.video, a.step, count=a.count))
        h, w = frames[0][2].shape[:2]
        fov, _ = models.focal_fov(frames[0][2], frames[min(len(frames) - 1, 5)][2])
        fov = fov or 65.0
        f = w / (2 * np.tan(np.radians(fov) / 2))
        cx, cy, B = (w - 1) / 2, (h - 1) / 2, None
        if a.mono == "vda":
            from stereoguard.mono import VideoDepthAnythingMetric
            vda = VideoDepthAnythingMetric(encoder="vitl", device=models.dev)
            video_depth = vda(np.stack([fr[2][..., ::-1] for fr in frames]))
            del vda
        else:
            moge = models.get("moge2")
            video_depth = [moge(fr[2], hfov_deg=fov)[0] for fr in frames]

    writer = None
    log = []
    gif_frames = []
    t_all = time.perf_counter()
    for k, (idx, t, L, R) in enumerate(frames):
        if a.scale != 1.0 and R is not None:
            L = cv2.resize(L, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
            R = cv2.resize(R, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        depth, conf, objs, fs = video.analyse_frame(L, R, f, cx, cy, B, models, prof,
                                                    depth=None if video_depth is None else video_depth[k])
        objs = video.update_objects(objs, hist, t, prof)
        dt = time.perf_counter() - t0
        ar = viz.draw_ar(L, objs, fs, prof, title=f"StereoGuard  t={t:5.1f}s", max_labels=4)
        frame = compose(ar, viz.draw_bev(fs, objs, prof)) if fs is not None else ar
        if writer is None:
            writer = cv2.VideoWriter(os.path.join(a.out, "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        gw = a.gif_width
        gif_frames.append(cv2.resize(frame, (gw, int(frame.shape[0] * gw / frame.shape[1])), interpolation=cv2.INTER_AREA))
        log.append({"frame": idx, "t": round(t, 3), "proc_s": round(dt, 2),
                    "objects": [{"id": o.det.track_id, "name": o.det.name, "z": round(o.dist_m, 2),
                                 "x": round(o.x_m, 2), "closing_mps": round(o.speed_mps, 2),
                                 "ttc_s": None if not np.isfinite(o.ttc_s) else round(o.ttc_s, 2),
                                 "ttc_scale_s": (None if not np.isfinite(o.extra.get("ttc_scale", np.inf))
                                                 else round(o.extra["ttc_scale"], 2)),
                                 "level": o.level, "in_path": o.in_path} for o in objs]})
        print(f"frame {idx:4d} t={t:5.2f}s {dt:5.2f}s  " +
              ", ".join(f"{o.det.name}#{o.det.track_id} {o.dist_m:.1f}m {o.level}" for o in objs[:4]))
    if writer is not None:
        writer.release()
    with open(os.path.join(a.out, "tracks.json"), "w") as fjson:
        json.dump(log, fjson, indent=1)
    if gif_frames:
        import imageio.v2 as imageio
        imageio.mimsave(os.path.join(a.out, "annotated.gif"), [g[..., ::-1] for g in gif_frames[::2]],
                        duration=0.2, loop=0)
    print(f"done: {len(log)} frames in {time.perf_counter() - t_all:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
