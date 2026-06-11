import argparse
import os
import cv2
import numpy as np


def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def resize_to_width(img, width=1280):
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / float(w)
    new_h = int(h * scale)
    return cv2.resize(img, (width, new_h), interpolation=cv2.INTER_AREA)


def add_title(img, title, subtitle=None):
    vis = img.copy()

    # dark title bar
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 90), (0, 0, 0), -1)

    cv2.putText(
        vis,
        title,
        (30, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if subtitle:
        cv2.putText(
            vis,
            subtitle,
            (30, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    return vis


def pad_to_same_size(img, target_w, target_h):
    h, w = img.shape[:2]
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x0 = (target_w - w) // 2
    y0 = (target_h - h) // 2

    canvas[y0:y0 + h, x0:x0 + w] = img
    return canvas


def make_slide(image_path, title, subtitle, output_width=1280):
    img = read_image(image_path)
    img = resize_to_width(img, output_width)
    img = add_title(img, title, subtitle)
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Make a demo video for stereo-based obstacle awareness and depth ordering."
    )
    parser.add_argument(
        "--result_dir",
        default="results/depth_order_scene1",
        help="folder containing output images from object_depth_order_demo.py",
    )
    parser.add_argument(
        "--output",
        default="depth_order_demo_video.mp4",
        help="output video path",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="video FPS. Use 1 for slide-style demo.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="output video width",
    )
    args = parser.parse_args()

    # Each entry: filename, title, subtitle, duration in seconds
    slides = [
        (
            "matches.png",
            "Step 1: Feature Matching",
            "Find corresponding points between the two phone images.",
            3,
        ),
        (
            "rectified_pair_guides.png",
            "Step 2: Stereo Rectification",
            "Horizontal guide lines show that corresponding points are aligned.",
            3,
        ),
        (
            "disparity_color.png",
            "Step 3: Disparity Map",
            "Larger disparity usually means the object is closer.",
            3,
        ),
        (
            "object_depth_order_result.png",
            "Step 4: Object Depth Ordering",
            "Objects are ranked by reliable p75 disparity score from near to far.",
            3,
        ),
        (
            "obstacle_warning_result.png",
            "Final Application: Obstacle Awareness",
            "Red: DANGER / TOO CLOSE, Yellow: CAUTION, Green: SAFE / FAR.",
            5,
        ),
    ]

    frames = []

    for filename, title, subtitle, duration in slides:
        path = os.path.join(args.result_dir, filename)

        # 如果沒有 obstacle_warning_result.png，就用 object_depth_order_result.png 代替
        if filename == "obstacle_warning_result.png" and not os.path.exists(path):
            path = os.path.join(args.result_dir, "object_depth_order_result.png")

        if not os.path.exists(path):
            print(f"[Warning] Missing {filename}, skip this slide.")
            continue

        slide = make_slide(path, title, subtitle, output_width=args.width)
        frames.append((slide, duration))

    if not frames:
        raise RuntimeError("No valid frames found. Please run object_depth_order_demo.py first.")

    # Make all frames same size
    max_w = max(frame.shape[1] for frame, _ in frames)
    max_h = max(frame.shape[0] for frame, _ in frames)

    normalized_frames = []
    for frame, duration in frames:
        padded = pad_to_same_size(frame, max_w, max_h)
        repeat = max(1, int(duration * args.fps))
        for _ in range(repeat):
            normalized_frames.append(padded)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps, (max_w, max_h))

    if not writer.isOpened():
        raise RuntimeError("Cannot open video writer. Try changing output path or codec.")

    for frame in normalized_frames:
        writer.write(frame)

    writer.release()

    print(f"Saved demo video to: {args.output}")
    print("Slides included:")
    for _, title, _, _ in slides:
        print(f"- {title}")


if __name__ == "__main__":
    main()
