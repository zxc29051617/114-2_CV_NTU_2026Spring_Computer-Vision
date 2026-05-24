import numpy as np
import cv2
import random
from pathlib import Path
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homography between consecutive images.
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]

    last_best_H = np.eye(3)
    out = dst

    for idx in tqdm(range(len(imgs) - 1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # 1. Feature detection and matching
        orb = cv2.ORB_create(nfeatures=3000, fastThreshold=7)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            matches = []
            im1_pts = np.empty((0, 2), dtype=np.float32)
            im2_pts = np.empty((0, 2), dtype=np.float32)
        else:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            raw_matches = matcher.knnMatch(des1, des2, k=2)

            matches = []
            for pair in raw_matches:
                if len(pair) == 2:
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        matches.append(m)

            matches = sorted(matches, key=lambda x: x.distance)[:1000]

            im1_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
            im2_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

        # 2. Estimate homography using RANSAC
        best_H = np.eye(3)
        best_inliers = None

        if len(matches) >= 4:
            max_iter = 500
            inlier_threshold = 5.0
            rng = np.random.default_rng(999 + idx)

            im2_homo = np.column_stack(
                (im2_pts, np.ones(len(im2_pts), dtype=np.float32))
            )

            for _ in range(max_iter):
                sample_idx = rng.choice(len(matches), 4, replace=False)

                H = solve_homography(im2_pts[sample_idx], im1_pts[sample_idx])

                if H is None or not np.isfinite(H).all():
                    continue

                projected = im2_homo @ H.T
                valid = np.abs(projected[:, 2]) > 1e-8

                projected[valid, :2] /= projected[valid, 2:3]

                errors = np.full(len(matches), np.inf)
                errors[valid] = np.linalg.norm(
                    projected[valid, :2] - im1_pts[valid], axis=1
                )

                inliers = errors < inlier_threshold

                if best_inliers is None or inliers.sum() > best_inliers.sum():
                    best_H = H
                    best_inliers = inliers

            if best_inliers is not None and best_inliers.sum() >= 4:
                best_H = solve_homography(
                    im2_pts[best_inliers],
                    im1_pts[best_inliers]
                )

        # 3. Chain homographies to the first frame coordinate
        best_H = last_best_H @ best_H

        if best_H[2, 2] != 0:
            best_H = best_H / best_H[2, 2]

        last_best_H = best_H

        # 4. Apply backward warping
        h2, w2 = im2.shape[:2]

        corners = np.array([
            [0, 0, 1],
            [w2, 0, 1],
            [w2, h2, 1],
            [0, h2, 1]
        ], dtype=np.float32)

        warped_corners = corners @ best_H.T
        warped_corners = warped_corners[:, :2] / warped_corners[:, 2:3]

        xmin = max(0, int(np.floor(warped_corners[:, 0].min())))
        xmax = min(w_max, int(np.ceil(warped_corners[:, 0].max())) + 1)
        ymin = max(0, int(np.floor(warped_corners[:, 1].min())))
        ymax = min(h_max, int(np.ceil(warped_corners[:, 1].max())) + 1)

        if xmin < xmax and ymin < ymax:
            dst = warping(im2, dst, best_H, ymin, ymax, xmin, xmax, direction='b')

        out = dst

    return out


if __name__ == "__main__":
    resource_dir = Path(__file__).resolve().parent.parent / "resource"
    FRAME_NUM = len(list(resource_dir.glob("frame*.jpg")))

    imgs = [
        cv2.imread(str(resource_dir / 'frame{:d}.jpg'.format(x)))
        for x in range(1, FRAME_NUM + 1)
    ]

    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)