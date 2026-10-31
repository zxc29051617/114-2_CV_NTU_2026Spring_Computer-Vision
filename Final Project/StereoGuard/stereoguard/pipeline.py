"""End-to-end analysis of one stereo pair.

    phone photos --(rectify.py)--> rectified pair --(stereo.py)--> disparity + left-right check
      --(fusion.py)--> metric depth (known slide length, or estimated from a monocular metric model)
      --(objects.py)--> named objects with distance / lateral offset / warning level
      --(freespace.py)--> support plane, top-view occupancy, suggested heading
"""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import camera, fusion, freespace, objects, rectify
from .stereo import Team18Census, shift_right


@dataclass
class Result:
    image: np.ndarray                 # rectified left view at processing resolution (BGR)
    right: np.ndarray
    f: float
    cx: float
    cy: float
    disp: np.ndarray
    confident: np.ndarray
    depth: np.ndarray                 # final metric depth (m)
    depth_stereo: np.ndarray
    mono_depth: np.ndarray = None
    baseline_m: float = float("nan")
    baseline_source: str = ""
    baseline_spread: float = float("nan")
    objects: list = field(default_factory=list)
    fs: object = None
    profile: object = None
    rect_stats: dict = field(default_factory=dict)
    intrinsics_source: str = ""
    timings: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    valid: np.ndarray = None          # pixels inside the photo after rectification
    scale_votes: list = field(default_factory=list)   # (model, slide length) when it was estimated
    R_left: np.ndarray = None         # rectifying rotation of the left camera (for ground-truth warping)


class Models:
    """Lazy holder so the Gradio app / CLI load every network only once.

    Defaults were chosen from the measurements in eval/ (see README):
      stereo  GPU: FoundationStereo ViT-L (most accurate)   CPU: Fast-FoundationStereo (ONNX)
      scale   GPU: seven single-image metric models (UniDepthV2-L, MoGe-2 S/L, Metric3D v2, Depth Pro,
                   DA-V2 Metric S/L), each bias-corrected and weighted by its log-error variance
                   (scale_fusion.json, fitted by eval/eval_scale_fusion.py; ~10 % error on held-out
                   phone pairs)   CPU: DA-V2 Metric-S with its bias correction
      focal   EXIF if present, otherwise GPU: VGGT (4 % error on phone-camera data)   CPU: MoGe-2
    """

    SCALE_GPU = ("unidepth", "moge2l", "metric3d", "moge2", "dav2l", "dav2s", "depthpro")
    SCALE_CPU = ("dav2s",)

    def __init__(self, device="auto", stereo="auto", detector="yolo11s-seg.pt", scale="auto", focal="auto",
                 prompts=None, det_conf=0.3, mono=None, stereo_device=None, low_vram=True):
        from .stereo import resolve_device
        self.dev = resolve_device(device, need_gb=5.0) if device == "auto" else device
        self.gpu = self.dev == "cuda"
        self.stereo_device = stereo_device or self.dev
        self.stereo_name = ("fs" if self.gpu else "ffs") if stereo == "auto" else stereo
        if mono is not None:                      # older option name: one model or none
            scale = mono or "none"
        if scale == "auto":
            self.scale_names = list(self.SCALE_GPU if self.gpu else self.SCALE_CPU)
        else:
            self.scale_names = [] if scale in (None, "none") else scale.split("+")
        self.focal_name = ("vggt+depthpro+moge2" if self.gpu else "moge2") if focal == "auto" else focal
        self.detector_name = detector
        self.prompts = prompts
        self.det_conf = det_conf
        self.low_vram = low_vram and self.gpu    # free each big network after use (the GPU is shared)
        self._cache = {}

    # ---------------------------------------------------------------- generic lazy loading
    def get(self, name):
        if name not in self._cache:
            self._cache[name] = self._build(name)
        return self._cache[name]

    def done(self, name):
        """Called after a network has been used for this pair."""
        if self.low_vram:
            self.release(name)

    def release(self, name):
        if self._cache.pop(name, None) is not None and self.gpu:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()

    def _build(self, name):
        from . import mono, multiview
        from .stereo import FastFoundationStereo, FoundationStereoTorch
        dev = self.dev
        builders = {
            "fs": lambda: FoundationStereoTorch(device=self.stereo_device),
            "ffs": lambda: FastFoundationStereo(device=self.stereo_device),
            "census": lambda: Team18Census(),
            "dav2s": lambda: mono.DepthAnythingV2Metric(size="Small", device=dev),
            "dav2l": lambda: mono.DepthAnythingV2Metric(size="Large", device=dev),
            "moge2": lambda: mono.MoGe2(device=dev),
            "depthpro": lambda: mono.DepthPro(device=dev),
            "unidepth": lambda: mono.UniDepthV2(device=dev),
            "metric3d": lambda: mono.Metric3Dv2(device=dev),
            "moge2l": lambda: mono.MoGe2(repo="Ruicheng/moge-2-vitl-normal", device=dev),
            "vggt": lambda: multiview.VGGTModel(device=dev),
            "mapanything": lambda: multiview.MapAnythingModel(device=dev),
        }
        return builders[name]()

    # ---------------------------------------------------------------- named roles
    @property
    def stereo(self):
        return self.get(self.stereo_name)

    @property
    def det(self):
        if "det" not in self._cache and self.detector_name:
            if self.detector_name == "gsam":
                self._cache["det"] = objects.GroundedSAM2(self.prompts or objects.DEFAULT_PROMPTS, device=self.dev)
            else:
                self._cache["det"] = objects.Detector(self.detector_name, conf=self.det_conf, device="cpu",
                                                      prompts=self.prompts)
        return self._cache.get("det")

    @property
    def mono(self):
        """First scale model: also used to fill pixels that fail the left-right check."""
        return self.get(self.scale_names[0]) if self.scale_names else None

    def focal_fov(self, img1, img2):
        """Horizontal field of view predicted from the photos (used when there is no EXIF).

        Several predictors joined by '+' are combined with the median: on phone-camera data VGGT alone
        was best (4 % focal error) but it failed on the Team 18 photos, the median of VGGT, Depth Pro and
        MoGe-2 was 7 % and robust to one model being wrong."""
        if self.focal_name in ("none", None):
            return None, "phone default"
        labels = {"vggt": "VGGT", "mapanything": "MapAnything", "moge2": "MoGe-2", "depthpro": "Depth Pro",
                  "unidepth": "UniDepthV2"}
        fovs = []
        for name in self.focal_name.split("+"):
            if name in ("vggt", "mapanything"):
                res = self.get(name)([img1, img2])
                f = float(res.K[0][0, 0])
                self.release(name)                    # 5 GB of weights: do not keep them around
                fov = float(np.degrees(2 * np.arctan(img1.shape[1] / (2 * f))))
            else:
                kw = {"max_side": 768} if name == "moge2" else {}
                _, fov = self.get(name)(img1, **kw)
                self.done(name)
            fovs.append((labels.get(name, name), fov))
        self.focal_votes = fovs
        fov = float(np.median([v for _, v in fovs]))
        label = "median of " + ", ".join(f"{n} {v:.1f}" for n, v in fovs) if len(fovs) > 1 else fovs[0][0]
        return fov, label

_FUSION = None


def fuse_slide_lengths(votes):
    """Combine (model, slide length) estimates: remove each model's mean log bias and average the logs
    with weights 1 / variance (both measured on Middlebury + ARKitScenes, stereoguard/scale_fusion.json).
    Returns (slide length, weighted spread of the corrected log estimates)."""
    global _FUSION
    if _FUSION is None:
        import json
        import os
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scale_fusion.json")
        _FUSION = json.load(open(f))["models"] if os.path.exists(f) else {}
    w = np.array([_FUSION.get(n, {}).get("weight", 1 / 0.3 ** 2) for n, _ in votes])
    lg = np.array([np.log(b) - _FUSION.get(n, {}).get("bias", 0.0) for n, b in votes])
    mean = float((w * lg).sum() / w.sum())
    spread = float(np.sqrt((w * (lg - mean) ** 2).sum() / w.sum())) if len(lg) > 1 else float("nan")
    return float(np.exp(mean)), spread


def _resize(img, w):
    if img.shape[1] == w:
        return img
    return cv2.resize(img, (w, int(round(img.shape[0] * w / img.shape[1]))), interpolation=cv2.INTER_AREA)


def analyse_rectified(left, right, f, cx, cy, models: Models, baseline_m=None, profile="tabletop",
                      use_mono=True, max_disp=None, timings=None, notes=None, hfov_known=True, d_off=0,
                      valid=None):
    """Core: rectified pair + focal length (+ optional baseline) -> depth, objects, free space.

    d_off: the right view is shifted by d_off pixels before matching (disparity offset).
    """
    timings = {} if timings is None else timings
    notes = [] if notes is None else notes
    prof = freespace.PROFILES[profile] if isinstance(profile, str) else profile
    h, w = left.shape[:2]

    t = time.perf_counter()
    st = models.stereo
    right_in = shift_right(right, d_off)
    if isinstance(st, Team18Census):
        st.max_disp = int(max_disp or 96)
        disp, reliable = st(left, right_in)
        confident = reliable > 0
    else:
        disp, disp_r = st(left, right_in, with_right=True)
        confident = fusion.lr_consistency(disp, disp_r)
    models.done(models.stereo_name)
    disp = disp + d_off
    valid_px = np.ones((h, w), bool) if valid is None else valid
    confident &= valid_px & (disp > 0.5)
    timings["stereo_s"] = time.perf_counter() - t

    mono_depth = None
    hfov = float(np.degrees(2 * np.arctan(w / (2 * f)))) if hfov_known else None
    b_list = []
    if use_mono and models.scale_names:
        t = time.perf_counter()
        for k, name in enumerate(models.scale_names):
            z_m, _ = models.get(name)(left, hfov_deg=hfov)
            models.done(name)
            if k == 0:
                mono_depth = z_m
            if not baseline_m:
                b, _ = fusion.estimate_baseline(disp, f, z_m, confident)
                if b:
                    b_list.append((name, b))
        timings["mono_s"] = time.perf_counter() - t

    spread = float("nan")
    if baseline_m:
        src = "given"
    else:
        if not b_list:
            raise ValueError("Unknown slide length: give --baseline_cm or enable a monocular metric model.")
        baseline_m, spread = fuse_slide_lengths(b_list)
        src = "estimated: " + ", ".join(f"{n} {100 * b:.1f} cm" for n, b in b_list)
    depth_stereo = fusion.disparity_to_depth(disp, f, baseline_m)
    depth_stereo[~valid_px] = np.nan
    depth, _ = fusion.fuse(depth_stereo, confident, mono_depth)
    depth[~valid_px] = np.nan

    t = time.perf_counter()
    fs = freespace.analyse(depth, f, cx, cy, prof)
    timings["freespace_s"] = time.perf_counter() - t
    if fs is None:
        notes.append("no support plane found: free-space hint disabled")

    objs = []
    if models.det is not None:
        t = time.perf_counter()
        dets = models.det.detect(left)
        objs = objects.measure_objects(dets, depth, f, cx, cy, prof, reliable=confident,
                                       plane=fs.plane if fs is not None else None)
        timings["detect_s"] = time.perf_counter() - t
        if src != "given":
            notes += objects.size_check(objs, left.shape)

    return Result(left, right, f, cx, cy, disp, confident, depth, depth_stereo, mono_depth, float(baseline_m),
                  src, spread, objs, fs, prof, timings=timings, notes=notes, valid=valid_px, scale_votes=b_list)


def analyse_phone_pair(img1, img2, models: Models, baseline_m=None, f35=None, path1=None, profile="tabletop",
                       work_width=1600, stereo_width=960, use_mono=True, K_full=None, pose=None):
    """Two photos from a sideways phone slide -> Result."""
    timings, notes = {}, []
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    W0 = img1.shape[1]
    a = _resize(img1, min(work_width, W0))
    b = _resize(img2, min(work_width, W0))
    h, w = a.shape[:2]

    hfov_model, fov_src = None, None
    if K_full is None and not f35 and camera.read_exif_f35(path1) is None:
        t = time.perf_counter()
        hfov_model, fov_src = models.focal_fov(a, b)
        timings["fov_s"] = time.perf_counter() - t
    K = camera.guess_intrinsics(path1, w, h, f35_user=f35, hfov_model=hfov_model, model_name=fov_src)
    if K_full is not None:                       # full calibration known (e.g. from the capture app)
        sk = w / W0
        K = camera.Intrinsics(K_full[0][0] * sk, K_full[1][1] * sk, (K_full[0][2] + 0.5) * sk - 0.5,
                              (K_full[1][2] + 0.5) * sk - 0.5, w, h, "given")

    t = time.perf_counter()
    rp = rectify.rectify_calibrated(a, b, K, pose=pose)
    timings["rectify_s"] = time.perf_counter() - t
    if rp.swapped:
        notes.append("the second photo was taken from the left: views swapped")
    if abs(rp.t[2]) > 0.35:
        notes.append(f"the phone also moved forward/backward (t_z = {rp.t[2]:.2f}); rectification may be poor")

    # choose the stereo resolution and a disparity offset so that the disparities of the matched
    # features fit inside the search range of the matcher (the original fixed max_disp = 96 did not)
    lim = models.stereo.range_limit
    s = min(stereo_width / w, 1.0)
    for _ in range(12):
        dmin, dmax = rp.stats["disp_p1"] * s, rp.stats["disp_p99"] * s
        d_off = 0 if dmax * 1.15 <= lim else int(max(0.0, np.floor(0.8 * dmin)))
        if (dmax - d_off) * 1.15 <= lim:
            break
        s *= 0.9
    L = cv2.resize(rp.left, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    R = cv2.resize(rp.right, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    f = rp.f * s
    cx, cy = (rp.cx + 0.5) * s - 0.5, (rp.cy + 0.5) * s - 0.5
    max_disp = int(np.ceil((dmax - d_off) * 1.15))
    valid = cv2.resize(rp.valid.astype(np.uint8), (L.shape[1], L.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0   # drop the interpolated rim
    res = analyse_rectified(L, R, f, cx, cy, models, baseline_m, profile, use_mono, max_disp, timings, notes,
                            d_off=d_off, valid=valid)
    res.rect_stats = dict(rp.stats)
    res.rect_stats.update({"stereo_scale": s, "stereo_width": L.shape[1], "disp_offset_px": d_off,
                           "feature_disp_range_px": [round(dmin, 1), round(dmax, 1)]})
    res.intrinsics_source = K.source
    res.R_left = rp.R_left
    res.R_rel_est = rp.R
    res.swapped = rp.swapped
    return res
