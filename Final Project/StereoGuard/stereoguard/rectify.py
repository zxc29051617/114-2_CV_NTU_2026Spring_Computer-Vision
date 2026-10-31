"""Rectify two photos taken by sliding one phone sideways.

The original bonus used ORB + fundamental matrix + cv2.stereoRectifyUncalibrated.  A projective
rectification is only defined up to a horizontal shear/offset, so disparities can become negative
(the background got about -60 px in scene1) and depth cannot be converted to meters.

Here the focal length is known (EXIF / model / prior), so we estimate the essential matrix,
recover the relative rotation R and the translation direction t, and apply a calibrated
(Fusiello-style) rectification with cv2.stereoRectify.  Disparity is then non-negative and
    Z = f * B / d
where B is the (metric) length of the slide.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import Intrinsics


@dataclass
class RectifiedPair:
    left: np.ndarray            # rectified left image (BGR)
    right: np.ndarray           # rectified right image (BGR)
    f: float                    # focal length of the rectified views (pixels)
    cx: float
    cy: float
    swapped: bool               # True if the second photo turned out to be the left view
    R: np.ndarray               # rotation between the two photos
    t: np.ndarray               # unit translation direction
    stats: dict = field(default_factory=dict)
    H_left: np.ndarray = None   # 3x3 homography: original left image -> rectified left
    pts_left: np.ndarray = None  # inlier matches in rectified coordinates (for diagnostics)
    pts_right: np.ndarray = None
    valid: np.ndarray = None    # pixels of the rectified left view that come from inside the photo
    R_left: np.ndarray = None   # rotation applied to the left camera by the rectification


def match_features(img1, img2, max_features=8000, ratio=0.8):
    """SIFT + ratio test + mutual nearest neighbour.  Returns Nx2 float arrays."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=max_features)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        raise RuntimeError("Not enough features: use a more textured scene or better light.")
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m12 = bf.knnMatch(d1, d2, k=2)
    m21 = {m[0].queryIdx: m[0].trainIdx for m in bf.knnMatch(d2, d1, k=1) if m}
    good = [p[0] for p in m12 if len(p) == 2 and p[0].distance < ratio * p[1].distance
            and m21.get(p[0].trainIdx) == p[0].queryIdx]
    if len(good) < 30:
        raise RuntimeError(f"Only {len(good)} good matches: the two photos overlap too little.")
    p1 = np.float64([k1[m.queryIdx].pt for m in good])
    p2 = np.float64([k2[m.trainIdx].pt for m in good])
    return p1, p2


def _apply_h(H, pts):
    p = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H)
    return p.reshape(-1, 2)


def rectification_stats(pl, pr):
    """Vertical error and horizontal disparity range of matched points after rectification."""
    dy = np.abs(pl[:, 1] - pr[:, 1])
    d = pl[:, 0] - pr[:, 0]
    return {
        "n_matches": int(len(pl)),
        "median_abs_dy_px": float(np.median(dy)),
        "p90_abs_dy_px": float(np.percentile(dy, 90)),
        "disp_p1": float(np.percentile(d, 1)),
        "disp_p50": float(np.percentile(d, 50)),
        "disp_p99": float(np.percentile(d, 99)),
        "negative_disp_ratio": float(np.mean(d < 0)),
    }


def rectify_calibrated(img1, img2, K: Intrinsics, alpha=0.5, match_width=1600, pose=None):
    """Calibrated rectification of a sideways phone slide.

    img1/img2 must have the same size as K.  Matching runs on a downscaled copy for speed.
    pose: optional known (R, t) with x2 = R x1 + t (e.g. from AR tracking) instead of the
    essential matrix; used to separate pose errors from matching errors in the evaluation.
    """
    h, w = img1.shape[:2]
    s = min(1.0, match_width / w)
    small1 = cv2.resize(img1, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img1
    small2 = cv2.resize(img2, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img2
    p1, p2 = match_features(small1, small2)
    p1 = (p1 + 0.5) / s - 0.5
    p2 = (p2 + 0.5) / s - 0.5

    Km = K.K
    E, inl = cv2.findEssentialMat(p1, p2, Km, method=cv2.USAC_MAGSAC, prob=0.9999, threshold=1.0 / s)
    if E is None:
        raise RuntimeError("Essential matrix estimation failed.")
    if E.shape[0] > 3:
        E = E[:3]
    inl = inl.ravel().astype(bool)
    _, R, t, pose_mask = cv2.recoverPose(E, p1[inl], p2[inl], Km)
    t = t.ravel()
    if pose is not None:
        R, t = np.asarray(pose[0], float), np.asarray(pose[1], float).ravel()
        t = t / np.linalg.norm(t)
    p1i, p2i = p1[inl], p2[inl]

    # camera-2 center in camera-1 coordinates; if it is on the left, swap the roles
    c2 = -R.T @ t
    swapped = bool(c2[0] < 0)
    if swapped:
        img1, img2 = img2, img1
        p1i, p2i = p2i, p1i
        R, t = R.T, -R.T @ t

    dist = np.zeros(5)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        Km, dist, Km, dist, (w, h), R, t.reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)
    m1 = cv2.initUndistortRectifyMap(Km, dist, R1, P1, (w, h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(Km, dist, R2, P2, (w, h), cv2.CV_32FC1)
    left = cv2.remap(img1, m1[0], m1[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    right = cv2.remap(img2, m2[0], m2[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    ones = np.full((h, w), 255, np.uint8)
    valid = cv2.remap(ones, m1[0], m1[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0

    # homographies (no lens distortion): x_rect ~ P[:, :3] R_i K^-1 x
    H1 = P1[:, :3] @ R1 @ np.linalg.inv(Km)
    H2 = P2[:, :3] @ R2 @ np.linalg.inv(Km)
    pl, pr = _apply_h(H1, p1i), _apply_h(H2, p2i)
    stats = rectification_stats(pl, pr)
    stats.update({
        "method": "calibrated (essential matrix + stereoRectify)",
        "inliers": int(inl.sum()),
        "rotation_deg": float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))),
        "t_dir": [float(v) for v in t / np.linalg.norm(t)],
    })
    stats["valid_area_ratio"] = float(valid.mean())
    return RectifiedPair(left, right, float(P1[0, 0]), float(P1[0, 2]), float(P1[1, 2]), swapped,
                         R, t / np.linalg.norm(t), stats, H1, pl, pr, valid, R1)


def rectify_uncalibrated_original(img1, img2):
    """The original Team 18 bonus rectification (ORB + F + stereoRectifyUncalibrated), for comparison."""
    h, w = img1.shape[:2]
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31)
    k1, d1 = orb.detectAndCompute(g1, None)
    k2, d2 = orb.detectAndCompute(g2, None)
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.5, 0.99)
    mask = mask.ravel().astype(bool)
    ok, H1, H2 = cv2.stereoRectifyUncalibrated(pts1[mask], pts2[mask], F, imgSize=(w, h), threshold=5)
    left = cv2.warpPerspective(img1, H1, (w, h))
    right = cv2.warpPerspective(img2, H2, (w, h))
    pl, pr = _apply_h(H1, pts1[mask]), _apply_h(H2, pts2[mask])
    stats = rectification_stats(pl, pr)
    stats.update({"method": "uncalibrated (original Team 18)", "inliers": int(mask.sum())})
    valid = cv2.warpPerspective(np.ones((h, w), np.uint8), H1, (w, h))
    stats["valid_area_ratio"] = float(valid.mean())
    return left, right, stats

