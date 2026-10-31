"""Shrink an annotated video into a small H.264 MP4 and a GIF for the README.

    python scripts/make_demo_media.py outputs/kitti_0005/annotated.mp4 results/kitti/kitti_demo
"""
import argparse

import cv2
import imageio.v2 as imageio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("out_prefix")
    p.add_argument("--mp4_width", type=int, default=960)
    p.add_argument("--gif_width", type=int, default=560)
    p.add_argument("--gif_step", type=int, default=2)
    p.add_argument("--fps", type=float, default=10)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    a = p.parse_args()
    cap = cv2.VideoCapture(a.video)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    frames = frames[a.start:a.end]

    def resize(f, w):
        return cv2.resize(f, (w, int(round(f.shape[0] * w / f.shape[1] / 2)) * 2), interpolation=cv2.INTER_AREA)

    w = imageio.get_writer(a.out_prefix + ".mp4", fps=a.fps, codec="libx264", quality=None,
                           ffmpeg_params=["-crf", "28", "-preset", "slow", "-pix_fmt", "yuv420p"])
    for f in frames:
        w.append_data(resize(f, a.mp4_width)[..., ::-1])
    w.close()
    gif = [resize(f, a.gif_width)[..., ::-1] for f in frames[::a.gif_step]]
    imageio.mimsave(a.out_prefix + ".gif", gif, duration=a.gif_step / a.fps, loop=0)
    print(f"{len(frames)} frames -> {a.out_prefix}.mp4 / .gif")


if __name__ == "__main__":
    main()
