"""Team 18 phone scene: the original bonus pipeline next to StereoGuard.

    python eval/eval_phone_scene.py [--baseline_cm X] [--device auto]

Original pipeline (re-run with the command from bonus_run_instruction.txt): ORB + fundamental
matrix + stereoRectifyUncalibrated, Census matcher with max_disp = 96, p75 score in fixed boxes.
Writes results/phone/{original_vs_stereoguard.jpg, dashboard.jpg, orbit.gif, phone_scene_report.json}.
"""
import argparse
import json
import os
import sys
import time

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stereoguard import pipeline, rectify, viz  # noqa: E402
from stereoguard.report import dashboard, orbit_frames, save_outputs  # noqa: E402
from stereoguard.stereo import load_team18_module  # noqa: E402

BONUS = os.path.join(os.path.dirname(ROOT), "Stereo Matching Team_Team18", "Stereo Matching_bonus")
SCENE = os.path.join(BONUS, "phone_data", "scene1")
BOXES = "Object_1:501,254,89,422;Object_2:311,582,134,72;Object_3:139,88,296,287"   # from bonus_run_instruction.txt


def run_original(img1, img2):
    """Re-run the original bonus steps and return its warning image and numbers."""
    sys.path.insert(0, BONUS)
    sys.dont_write_bytecode = True          # do not leave __pycache__ in the original folder
    import object_depth_order_demo as orig  # the original script, imported unchanged
    s0 = 1200 / img1.shape[1]
    a = cv2.resize(img1, None, fx=s0, fy=s0, interpolation=cv2.INTER_AREA)
    b = cv2.resize(img2, None, fx=s0, fy=s0, interpolation=cv2.INTER_AREA)
    _, _, st = rectify.rectify_uncalibrated_original(a, b)
    gl, gr = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    _, _, _, p1, p2 = orig.detect_and_match(gl, gr)
    rl, rr, *_ = orig.rectify_uncalibrated(a, b, p1, p2)
    team18 = load_team18_module()
    t = time.perf_counter()
    d_final, d_rel = team18.computeDispWithConfidence(rl, rr, 96, confidence_margin=1.0)
    t_disp = time.perf_counter() - t
    results = orig.rank_boxes_by_disparity(d_rel, orig.parse_boxes(BOXES), score_percentile=75, min_valid_ratio=0.005)
    vis = orig.draw_depth_order(rl, results)
    info = {
        "rectification": st,
        "disparity_time_s": t_disp,
        "reliable_pixel_ratio": float((d_rel > 0).mean()),
        "ranking": [{"name": r["name"], "score": r["depth_score"], "reliable_ratio": r["valid_ratio"],
                     "label": ["DANGER", "CAUTION", "SAFE"][i] if i < 3 else "LOW"} for i, r in enumerate(results)],
    }
    disp_col = cv2.applyColorMap(cv2.normalize(d_final, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_TURBO)
    return vis, disp_col, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left", default=os.path.join(SCENE, "left.jpg"))
    p.add_argument("--right", default=os.path.join(SCENE, "right.jpg"))
    p.add_argument("--baseline_cm", type=float, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "phone"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    img1, img2 = cv2.imread(a.left), cv2.imread(a.right)

    vis_o, disp_o, info_o = run_original(img1, img2)

    models = pipeline.Models(device=a.device)
    t = time.perf_counter()
    res = pipeline.analyse_phone_pair(img1, img2, models, baseline_m=a.baseline_cm / 100 if a.baseline_cm else None,
                                      path1=a.left, profile="tabletop")
    total = time.perf_counter() - t
    summ = save_outputs(res, os.path.join(ROOT, "outputs", "phone_scene1"))
    ar = viz.draw_ar(res.image, res.objects, res.fs, res.profile)
    depth = viz.colorize_depth(res.depth)

    top = viz.hstack_pad([viz.caption(vis_o, "Team 18 bonus (re-run): manual boxes, rank-only labels"),
                          viz.caption(ar, "StereoGuard: detected objects, meters, real thresholds")], h=560)
    bottom = viz.hstack_pad([viz.caption(disp_o, "Team 18 disparity (max_disp 96, uncalibrated rectification)"),
                             viz.caption(depth, "StereoGuard metric depth (FoundationStereo, scale from three monocular models)")],
                            h=560)
    cv2.imwrite(os.path.join(a.out, "original_vs_stereoguard.jpg"), viz.vstack_pad([top, bottom], w=1600),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(os.path.join(a.out, "dashboard.jpg"), dashboard(res), [cv2.IMWRITE_JPEG_QUALITY, 85])
    cv2.imwrite(os.path.join(a.out, "ar.jpg"), ar, [cv2.IMWRITE_JPEG_QUALITY, 88])
    frames = orbit_frames(res, n=24, size=(560, 420))
    import imageio.v2 as imageio
    imageio.mimsave(os.path.join(a.out, "orbit.gif"), [f[..., ::-1] for f in frames], duration=0.12, loop=0)

    report = {"original": info_o, "stereoguard": summ, "stereoguard_total_s": total}
    with open(os.path.join(a.out, "phone_scene_report.json"), "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
