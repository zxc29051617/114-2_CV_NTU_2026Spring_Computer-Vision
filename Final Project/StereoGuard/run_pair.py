"""Analyse two phone photos taken by sliding the phone sideways.

    python run_pair.py --left data/phone/left.jpg --right data/phone/right.jpg --out outputs/scene1
    python run_pair.py ... --baseline_cm 6          # if you measured how far the phone moved
    python run_pair.py ... --profile walking         # thresholds for walking instead of a desk scene
    python run_pair.py ... --stereo census           # use the original Team 18 matcher instead
    python run_pair.py ... --detector gsam --prompts "bottle, book, monitor, cable, lamp"   # open vocabulary
"""
import argparse
import json

import cv2

from stereoguard import freespace, pipeline
from stereoguard.report import save_outputs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--out", default="outputs/pair")
    p.add_argument("--baseline_cm", type=float, default=None, help="length of the phone slide; estimated if omitted")
    p.add_argument("--f35", type=float, default=None, help="35 mm-equivalent focal length if EXIF is missing")
    p.add_argument("--profile", default="tabletop", choices=list(freespace.PROFILES))
    p.add_argument("--stereo", default="auto", choices=["auto", "fs", "ffs", "census"],
                   help="auto: FoundationStereo on a GPU, Fast-FoundationStereo on a CPU")
    p.add_argument("--scale", default="auto",
                   help="models for the slide length, joined by '+' (depthpro, unidepth, dav2l, dav2s, moge2)")
    p.add_argument("--focal", default="auto", help="field of view without EXIF: vggt, depthpro, moge2, none")
    p.add_argument("--detector", default="yolo11s-seg.pt",
                   help="yolo11s-seg.pt (COCO), gsam (Grounding DINO + SAM 2.1) or yoloe-26s-seg.pt; both need --prompts")
    p.add_argument("--prompts", default=None, help="comma separated object names for the open-vocabulary detectors")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default="auto", help="auto: GPU if >= 5 GB of it is free")
    p.add_argument("--stereo_width", type=int, default=960)
    a = p.parse_args()

    models = pipeline.Models(device=a.device, stereo=a.stereo, detector=a.detector, scale=a.scale, focal=a.focal,
                             prompts=a.prompts.split(",") if a.prompts else None, det_conf=a.conf)
    img1, img2 = cv2.imread(a.left), cv2.imread(a.right)
    if img1 is None or img2 is None:
        raise FileNotFoundError("cannot read the input photos")
    res = pipeline.analyse_phone_pair(img1, img2, models, baseline_m=a.baseline_cm / 100 if a.baseline_cm else None,
                                      f35=a.f35, path1=a.left, profile=a.profile, stereo_width=a.stereo_width)
    summary = save_outputs(res, a.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
