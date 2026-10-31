"""Download the public datasets used for evaluation.

    python scripts/download_data.py course        # Tsukuba / Venus / Teddy / Cones (course test pairs)
    python scripts/download_data.py middeval3     # Middlebury 2014 (MiddEval3) training set, quarter resolution
    python scripts/download_data.py mb2021        # Middlebury 2021 mobile stereo (phone camera, metric GT)
    python scripts/download_data.py kitti         # one short KITTI raw sequence (stereo video + LiDAR)
    python scripts/download_data.py all

Everything goes to data/ (ignored by git).
"""
import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

MB01 = "https://vision.middlebury.edu/stereo/data/scenes2001/data"
MB03 = "https://vision.middlebury.edu/stereo/data/scenes2003/newdata"
COURSE = {
    # name: (left url, right url, gt url)
    "Tsukuba": (f"{MB01}/tsukuba/scene1.row3.col3.ppm", f"{MB01}/tsukuba/scene1.row3.col4.ppm",
                f"{MB01}/tsukuba/truedisp.row3.col3.pgm"),
    "Venus": (f"{MB01}/venus/im2.ppm", f"{MB01}/venus/im6.ppm", f"{MB01}/venus/disp2.pgm"),
    "Teddy": (f"{MB03}/teddy/im2.png", f"{MB03}/teddy/im6.png", f"{MB03}/teddy/disp2.png"),
    "Cones": (f"{MB03}/cones/im2.png", f"{MB03}/cones/im6.png", f"{MB03}/cones/disp2.png"),
}

MIDDEVAL3 = [
    "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip",
    "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip",
]

MB21 = "https://vision.middlebury.edu/stereo/data/scenes2021/data"
MB21_SCENES = ["artroom1", "artroom2", "bandsaw1", "bandsaw2", "chess1", "chess2", "chess3",
               "curule1", "curule2", "curule3", "ladder1", "ladder2", "octogons1", "octogons2",
               "pendulum1", "pendulum2", "podium1", "skates1", "skates2", "skiboots1", "skiboots2",
               "skiboots3", "traproom1", "traproom2"]

KITTI_SEQ = "2011_09_26_drive_0005"
KITTI_URLS = [
    f"https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/{KITTI_SEQ}/{KITTI_SEQ}_sync.zip",
    "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_calib.zip",
]


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                sys.stdout.write(f"\r    {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
                sys.stdout.flush()
    if total:
        print()
    os.replace(tmp, dst)
    return dst


def unzip(path, out_dir, members=None):
    with zipfile.ZipFile(path) as z:
        for m in z.namelist():
            if members is not None and not any(s in m for s in members):
                continue
            # refuse absolute paths / parent references inside archives
            target = os.path.normpath(os.path.join(out_dir, m))
            if not target.startswith(os.path.normpath(out_dir) + os.sep):
                continue
            if m.endswith("/"):
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def get_course():
    import cv2
    out = os.path.join(DATA, "course")
    for name, urls in COURSE.items():
        d = os.path.join(out, name)
        raw = os.path.join(d, "raw")
        files = [fetch(u, os.path.join(raw, os.path.basename(u))) for u in urls]
        for src, dst in zip(files, ["img_left.png", "img_right.png", "disp_gt.png"]):
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(os.path.join(d, dst), img)
        print(f"course/{name}: ok")


def get_middeval3():
    out = os.path.join(DATA, "middeval3")
    for u in MIDDEVAL3:
        z = fetch(u, os.path.join(out, "zip", os.path.basename(u)))
        unzip(z, out)
    print("middeval3: ok")


def get_mb2021(scenes=None):
    out = os.path.join(DATA, "mb2021")
    for s in scenes or MB21_SCENES:
        for f in ["im0.png", "im1.png", "disp0.pfm", "calib.txt"]:
            fetch(f"{MB21}/{s}/{f}", os.path.join(out, s, f))
        print(f"mb2021/{s}: ok")


def get_kitti():
    out = os.path.join(DATA, "kitti_raw")
    for u in KITTI_URLS:
        z = fetch(u, os.path.join(out, "zip", os.path.basename(u)))
        # keep only the color stereo cameras, LiDAR, tracklets and calibration
        unzip(z, out, members=["image_02", "image_03", "velodyne_points", "calib_", "timestamps"])
    print("kitti: ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", nargs="+", choices=["course", "middeval3", "mb2021", "kitti", "all"])
    p.add_argument("--mb2021_scenes", nargs="*", default=None, help="subset of Middlebury 2021 scenes")
    a = p.parse_args()
    todo = set(a.what)
    if "all" in todo:
        todo = {"course", "middeval3", "mb2021", "kitti"}
    if "course" in todo:
        get_course()
    if "middeval3" in todo:
        get_middeval3()
    if "mb2021" in todo:
        get_mb2021(a.mb2021_scenes)
    if "kitti" in todo:
        get_kitti()


if __name__ == "__main__":
    main()
