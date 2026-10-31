"""Phone-camera stereo pairs with metric ground truth from ARKitScenes (Apple, iPad Pro + Faro laser scans).

    python scripts/download_arkitscenes.py --videos 12

For each validation video of the depth-upsampling subset (upright videos, one per visit):
  * the upsampling zip (~50 MB) gives ~20 frames with laser-scanner depth (highres_depth, mm)
  * lowres_wide.traj gives the ARKit camera poses, vga_wide_intrinsics the intrinsics
  * for every ground-truth frame the script looks along the video for a second frame that is a
    sideways slide like the Team 18 capture (8-30 cm, mostly along the camera x axis, < 5 deg rotation,
    second camera to the right) and fetches just those two 640x480 frames from the 2 GB vga_wide zip
    with HTTP range requests.
Output: data/arkitscenes/<video>_<t>/{left.png,right.png,depth_gt.png (mm, left view),K.txt,pair.json}
The ARKitScenes data is free for non-commercial use (see its LICENSE); it is not redistributed here.
"""
import argparse
import csv
import io
import json
import os
import re
import urllib.request
import zipfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "arkitscenes")
URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"


class HttpFile(io.RawIOBase):
    """Read-only, seekable view of a remote file through HTTP range requests (for zipfile)."""

    def __init__(self, url, block=1 << 20):
        self.url, self.pos, self.block, self.cache = url, 0, block, {}
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _blockdata(self, k):
        if k not in self.cache:
            a, b = k * self.block, min(self.size, (k + 1) * self.block) - 1
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                self.cache[k] = r.read()
            if len(self.cache) > 64:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[k]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while n > 0:
            k, o = divmod(self.pos, self.block)
            chunk = self._blockdata(k)[o:o + n]
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url), timeout=120) as r:
        return r.read()


def world_to_cam(traj, t):
    """World-to-camera pose at time t, interpolated between the 10 Hz ARKit samples
    (linear in translation, slerp in rotation).  Returns (R, t, distance to the nearest sample)."""
    from scipy.spatial.transform import Rotation, Slerp
    i = int(np.clip(np.searchsorted(traj[:, 0], t), 1, len(traj) - 1))
    t0, t1 = traj[i - 1, 0], traj[i, 0]
    a = float(np.clip((t - t0) / max(t1 - t0, 1e-9), 0.0, 1.0))
    rots = Rotation.from_rotvec(traj[i - 1:i + 1, 1:4])
    R = Slerp([0.0, 1.0], rots)([a]).as_matrix()[0]
    tr = (1 - a) * traj[i - 1, 4:7] + a * traj[i, 4:7]
    return R, tr, min(abs(t - t0), abs(t - t1)) if not (t0 <= t <= t1) else 0.0


def relative(p1, p2):
    """x2 = R x1 + t from world-to-camera poses."""
    R1, t1 = p1
    R2, t2 = p2
    R = R2 @ R1.T
    return R, t2 - R @ t1


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--videos", type=int, default=12)
    a.add_argument("--min_cm", type=float, default=8.0)
    a.add_argument("--max_cm", type=float, default=30.0)
    a.add_argument("--per_video", type=int, default=2)
    args = a.parse_args()
    os.makedirs(OUT, exist_ok=True)
    meta = list(csv.DictReader(io.StringIO(fetch(f"{URL}/upsampling/metadata.csv").decode())))
    vids, visits = [], set()
    for m in meta:
        if m["fold"] == "Validation" and m["sky_direction"] == "Up" and m["visit_id"] not in visits:
            vids.append(m["video_id"])
            visits.add(m["visit_id"])
    n_pairs = 0
    for vid in vids[:args.videos]:
        try:
            up = zipfile.ZipFile(io.BytesIO(fetch(f"{URL}/upsampling/Validation/{vid}.zip")))
            traj = np.loadtxt(io.BytesIO(fetch(f"{URL}/raw/Validation/{vid}/lowres_wide.traj")))
            kz = zipfile.ZipFile(io.BytesIO(fetch(f"{URL}/raw/Validation/{vid}/vga_wide_intrinsics.zip")))
            vga = zipfile.ZipFile(io.BufferedReader(HttpFile(f"{URL}/raw/Validation/{vid}/vga_wide.zip"), 1 << 20))
        except Exception as e:  # some videos lack an asset
            print(f"{vid}: skipped ({e})")
            continue
        vga_names = {float(re.search(r"_(\d+\.\d+)\.png$", n).group(1)): n for n in vga.namelist() if n.endswith(".png")}
        vga_ts = np.array(sorted(vga_names))
        k_names = {float(re.search(r"_(\d+\.\d+)\.pincam$", n).group(1)): n for n in kz.namelist() if n.endswith(".pincam")}
        gts = sorted(float(re.search(r"_(\d+\.\d+)\.png$", n).group(1)) for n in up.namelist() if "/highres_depth/" in n and n.endswith(".png"))
        used = []
        for t0 in gts:
            if any(abs(t0 - u) < 5 for u in used):
                continue
            p0 = world_to_cam(traj, t0)
            if p0[2] > 0.06:
                continue
            best = None
            for t1 in vga_ts[(vga_ts > t0 - 2.0) & (vga_ts < t0 + 2.0)]:
                p1 = world_to_cam(traj, t1)
                if p1[2] > 0.06:
                    continue
                R, t = relative(p0[:2], p1[:2])
                c2 = -R.T @ t                                # second camera centre in the first frame
                B = np.linalg.norm(c2)
                rot = np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))
                if not (args.min_cm / 100 <= B <= args.max_cm / 100) or rot > 5.0 or c2[0] < 0.85 * B:
                    continue
                score = abs(B - 0.15) + 0.01 * rot
                if best is None or score < best[0]:
                    best = (score, t1, B, rot, c2, R, t)
            t_near = min(vga_names, key=lambda x: abs(x - t0))
            if best is None or abs(t_near - t0) > 0.02:
                continue
            _, t1, B, rot, c2, Rr, tr = best
            name = f"{vid}_{t0:.3f}"
            d = os.path.join(OUT, name)
            os.makedirs(d, exist_ok=True)
            left = cv2.imdecode(np.frombuffer(vga.read(vga_names[t_near]), np.uint8), cv2.IMREAD_COLOR)
            right = cv2.imdecode(np.frombuffer(vga.read(vga_names[t1]), np.uint8), cv2.IMREAD_COLOR)
            gt_name = [n for n in up.namelist() if "/highres_depth/" in n and f"_{t0:.3f}.png" in n][0]
            gt = cv2.imdecode(np.frombuffer(up.read(gt_name), np.uint8), cv2.IMREAD_UNCHANGED)
            w, h, fx, fy, cx, cy = np.loadtxt(io.BytesIO(kz.read(k_names[min(k_names, key=lambda x: abs(x - t0))])))
            cv2.imwrite(os.path.join(d, "left.png"), left)
            cv2.imwrite(os.path.join(d, "right.png"), right)
            cv2.imwrite(os.path.join(d, "depth_gt.png"), gt)                 # 1920x1440, mm
            np.savetxt(os.path.join(d, "K.txt"), [[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            json.dump({"video": vid, "t_left": t0, "t_right": float(t1), "baseline_m": float(B),
                       "rotation_deg": float(rot), "c2_in_left": [float(v) for v in c2],
                       "R_rel": Rr.tolist(), "t_rel": tr.tolist()},
                      open(os.path.join(d, "pair.json"), "w"), indent=1)
            used.append(t0)
            n_pairs += 1
            print(f"{name}: slide {100 * B:.1f} cm, rotation {rot:.1f} deg, dt {t1 - t0:+.2f} s", flush=True)
            if len(used) >= args.per_video:
                break
    print(f"{n_pairs} pairs in {OUT}")


if __name__ == "__main__":
    main()
