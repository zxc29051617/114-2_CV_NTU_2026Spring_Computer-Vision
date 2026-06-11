import argparse
import os
from pathlib import Path
import cv2
import numpy as np


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def resize_for_processing(img, max_width=1200):
    h, w = img.shape[:2]
    if w <= max_width:
        return img, 1.0
    scale = max_width / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA), scale


def detect_and_match(left_gray, right_gray, max_features=6000):
    orb = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31)
    kp1, des1 = orb.detectAndCompute(left_gray, None)
    kp2, des2 = orb.detectAndCompute(right_gray, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        raise RuntimeError("Not enough ORB features. Try a more textured scene or better lighting.")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 12:
        raise RuntimeError(f"Too few good matches ({len(good)}). Retake images with more overlap and texture.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return kp1, kp2, good, pts1, pts2


def rectify_uncalibrated(left, right, pts1, pts2):
    h, w = left.shape[:2]
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.5, 0.99)
    if F is None or mask is None:
        raise RuntimeError("Fundamental matrix estimation failed. Retake the stereo pair.")
    mask = mask.ravel().astype(bool)
    in1 = pts1[mask]
    in2 = pts2[mask]
    if len(in1) < 8:
        raise RuntimeError(f"Too few RANSAC inliers ({len(in1)}). Retake images with less rotation and more overlap.")

    ok, H1, H2 = cv2.stereoRectifyUncalibrated(in1, in2, F, imgSize=(w, h), threshold=5)
    if not ok:
        raise RuntimeError("stereoRectifyUncalibrated failed. Retake images with mostly horizontal translation.")

    rect_l = cv2.warpPerspective(left, H1, (w, h))
    rect_r = cv2.warpPerspective(right, H2, (w, h))
    return rect_l, rect_r, F, mask, H1, H2


def draw_horizontal_guides(left, right, step=60):
    pair = np.hstack([left, right])
    h, w = pair.shape[:2]
    for y in range(0, h, step):
        cv2.line(pair, (0, y), (w - 1, y), (0, 255, 0), 1)
    return pair


def normalize_disparity(disp):
    disp_f = disp.astype(np.float32)
    valid = disp_f > 0
    if not np.any(valid):
        return np.zeros_like(disp, dtype=np.uint8)
    lo = np.percentile(disp_f[valid], 2)
    hi = np.percentile(disp_f[valid], 98)
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((disp_f - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return norm


def parse_boxes(box_args):
    # Format: name:x,y,w,h ; name:x,y,w,h
    boxes = []
    if not box_args:
        return boxes
    for item in box_args.split(';'):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError("Box format should be name:x,y,w,h")
        name, coords = item.split(':', 1)
        vals = [int(v.strip()) for v in coords.split(',')]
        if len(vals) != 4:
            raise ValueError("Box format should be name:x,y,w,h")
        boxes.append((name.strip(), tuple(vals)))
    return boxes


def select_rois_interactive(image):
    preview = image.copy()
    max_w = 1200
    h, w = preview.shape[:2]
    scale = 1.0
    if w > max_w:
        scale = max_w / w
        preview = cv2.resize(preview, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    print("Draw ROIs around objects from near/middle/far. Press ENTER/SPACE when done, ESC to cancel.")
    rois_small = cv2.selectROIs("Select objects on rectified left image", preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select objects on rectified left image")
    boxes = []
    for i, (x, y, bw, bh) in enumerate(rois_small):
        if bw <= 0 or bh <= 0:
            continue
        x = int(round(x / scale)); y = int(round(y / scale))
        bw = int(round(bw / scale)); bh = int(round(bh / scale))
        boxes.append((f"Object_{i+1}", (x, y, bw, bh)))
    return boxes


def rank_boxes_by_disparity(disp, boxes, score_percentile=75, min_valid_ratio=0.005):
    results = []
    h, w = disp.shape[:2]
    for name, (x, y, bw, bh) in boxes:
        x1 = max(0, x); y1 = max(0, y)
        x2 = min(w, x + bw); y2 = min(h, y + bh)
        if x2 <= x1 or y2 <= y1:
            continue

        patch = disp[y1:y2, x1:x2].astype(np.float32)
        valid = patch[patch > 0]
        valid_ratio = float(valid.size) / float(patch.size) if patch.size > 0 else 0.0

        if valid.size == 0:
            score = 0.0
            p50 = 0.0
            p75 = 0.0
        else:
            p50 = float(np.percentile(valid, 50))
            p75 = float(np.percentile(valid, 75))
            score = float(np.percentile(valid, score_percentile))
            if valid_ratio < min_valid_ratio:
                score = 0.0

        results.append({
            "name": name,
            "box": (x1, y1, x2 - x1, y2 - y1),
            "depth_score": score,
            "p50_disparity": p50,
            "p75_disparity": p75,
            "valid_ratio": valid_ratio,
            "valid_pixels": int(valid.size),
            "used_for_score": valid_ratio >= min_valid_ratio,
        })

    results.sort(key=lambda r: r["depth_score"], reverse=True)
    return results


def draw_depth_order(image, results):
    vis = image.copy()

    # rank 1: closest, rank 2: middle, rank 3: farthest
    risk_styles = [
        {
            "depth": "NEAR",
            "risk": "DANGER / TOO CLOSE",
            "color": (0, 0, 255),       # red in BGR
        },
        {
            "depth": "MIDDLE",
            "risk": "CAUTION / MID RANGE",
            "color": (0, 255, 255),     # yellow in BGR
        },
        {
            "depth": "FAR",
            "risk": "SAFE / FAR",
            "color": (0, 255, 0),       # green in BGR
        },
    ]

    for idx, r in enumerate(results):
        x, y, w, h = r["box"]

        if idx < len(risk_styles):
            style = risk_styles[idx]
            depth_label = style["depth"]
            risk_label = style["risk"]
            color = style["color"]
        else:
            depth_label = f"RANK {idx + 1}"
            risk_label = "LOW PRIORITY"
            color = (255, 255, 255)

        # Example: 1. Box - DANGER / TOO CLOSE (score=42.0)
        text = f"{idx+1}. {r['name']} - {risk_label} (score={r['depth_score']:.1f})"

        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 4)

        ty = max(35, y - 12)
        cv2.putText(
            vis,
            text,
            (x, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

        # Put a small filled label background inside the box
        label_inside = f"{depth_label}"
        cv2.rectangle(vis, (x, y), (x + 180, y + 35), color, -1)
        cv2.putText(
            vis,
            label_inside,
            (x + 8, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # Title
    title1 = "Stereo-based Obstacle Awareness"
    title2 = "Risk level is estimated from reliable p75 disparity: larger disparity = closer object"

    cv2.putText(vis, title1, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(vis, title1, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.putText(vis, title2, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, title2, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

    return vis


def save_csv(results, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('rank,name,depth_score,p50_disparity,p75_disparity,valid_ratio,valid_pixels,used_for_score,x,y,w,h\n')
        for i, r in enumerate(results, start=1):
            x, y, w, h = r['box']
            f.write(
                f"{i},{r['name']},{r['depth_score']:.4f},"
                f"{r['p50_disparity']:.4f},{r['p75_disparity']:.4f},"
                f"{r['valid_ratio']:.4f},{r['valid_pixels']},"
                f"{int(r['used_for_score'])},{x},{y},{w},{h}\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Stereo-based object depth ordering demo using one phone stereo pair.")
    parser.add_argument('--left', required=True, help='path to left image')
    parser.add_argument('--right', required=True, help='path to right image')
    parser.add_argument('--out_dir', default='results/depth_order_scene1')
    parser.add_argument('--max_disp', type=int, default=96)
    parser.add_argument('--score_percentile', type=float, default=75.0,
                        help='ROI disparity percentile used for depth ranking. Default: 75')
    parser.add_argument('--confidence_margin', type=float, default=1.0,
                        help='minimum winner cost margin for reliable ROI scoring. Default: 1.0')
    parser.add_argument('--min_reliable_ratio', type=float, default=0.005,
                        help='minimum reliable-pixel ratio required to use an ROI score. Default: 0.005')
    parser.add_argument('--boxes', default='', help='optional boxes: name:x,y,w,h;name:x,y,w,h')
    parser.add_argument('--no_interactive', action='store_true', help='do not open ROI selector; requires --boxes')
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    left = cv2.imread(args.left)
    right = cv2.imread(args.right)
    if left is None:
        raise FileNotFoundError(args.left)
    if right is None:
        raise FileNotFoundError(args.right)

    left, s1 = resize_for_processing(left)
    right, s2 = resize_for_processing(right)
    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)

    gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    kp1, kp2, good, pts1, pts2 = detect_and_match(gray_l, gray_r)
    match_img = cv2.drawMatches(left, kp1, right, kp2, good[:120], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(os.path.join(args.out_dir, 'matches.png'), match_img)

    rect_l, rect_r, F, mask, H1, H2 = rectify_uncalibrated(left, right, pts1, pts2)
    cv2.imwrite(os.path.join(args.out_dir, 'rectified_left.png'), rect_l)
    cv2.imwrite(os.path.join(args.out_dir, 'rectified_right.png'), rect_r)
    cv2.imwrite(os.path.join(args.out_dir, 'rectified_pair_guides.png'), draw_horizontal_guides(rect_l, rect_r))

    from computeDisp import computeDispWithConfidence
    disp, score_disp = computeDispWithConfidence(
        rect_l,
        rect_r,
        args.max_disp,
        confidence_margin=args.confidence_margin,
    )
    cv2.imwrite(os.path.join(args.out_dir, 'disparity_raw.png'), disp)
    disp_norm = normalize_disparity(disp)
    disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(args.out_dir, 'disparity_color.png'), disp_color)
    cv2.imwrite(os.path.join(args.out_dir, 'disparity_reliable_raw.png'), score_disp)
    score_disp_norm = normalize_disparity(score_disp)
    score_disp_color = cv2.applyColorMap(score_disp_norm, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(args.out_dir, 'disparity_reliable_color.png'), score_disp_color)

    boxes = parse_boxes(args.boxes)
    if not boxes and not args.no_interactive:
        boxes = select_rois_interactive(rect_l)
    if not boxes:
        raise RuntimeError("No object boxes selected. Use interactive selection or pass --boxes.")

    results = rank_boxes_by_disparity(
        score_disp,
        boxes,
        score_percentile=args.score_percentile,
        min_valid_ratio=args.min_reliable_ratio,
    )
    vis = draw_depth_order(rect_l, results)
    cv2.imwrite(os.path.join(args.out_dir, 'object_depth_order_result.png'), vis)
    cv2.imwrite(os.path.join(args.out_dir, 'obstacle_warning_result.png'), vis)
    save_csv(results, os.path.join(args.out_dir, 'depth_order.csv'))

    print(f"Depth order, near to far (score = reliable p{args.score_percentile:g} disparity):")
    for i, r in enumerate(results, start=1):
        print(
            f"{i}. {r['name']}: depth score = {r['depth_score']:.2f}, "
            f"p50 = {r['p50_disparity']:.2f}, p75 = {r['p75_disparity']:.2f}, "
            f"valid = {r['valid_ratio'] * 100:.1f}%"
        )
        if not r['used_for_score']:
            print("   Warning: too few reliable disparity pixels. This ROI score was set to 0.")
        elif r['valid_ratio'] < 0.05:
            print("   Warning: low reliable disparity ratio. This ROI may be textureless, reflective, transparent, or too large.")
    print(f"Saved results to: {args.out_dir}")


if __name__ == '__main__':
    main()
