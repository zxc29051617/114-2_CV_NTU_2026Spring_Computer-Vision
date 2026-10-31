"""Object detection / instance segmentation and per-object distance + warning level."""
import contextlib
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .freespace import Profile, ground_frame


@dataclass
class Detection:
    name: str
    conf: float
    box: tuple                 # x1, y1, x2, y2 (pixels)
    mask: np.ndarray           # HxW bool
    track_id: int = -1


@dataclass
class ObjectInfo:
    det: Detection
    dist_m: float              # near-surface depth (20th percentile of Z inside the mask)
    median_m: float
    x_m: float                 # lateral offset of the object centre (+ right)
    x_lo: float
    x_hi: float
    height_m: float = float("nan")   # height above the support plane (if found)
    width_m: float = float("nan")
    valid_ratio: float = 0.0   # share of mask pixels with a reliable stereo match
    in_path: bool = False
    level: str = "SAFE"
    ttc_s: float = float("inf")
    speed_mps: float = 0.0
    extra: dict = field(default_factory=dict)


LEVEL_COLORS = {"DANGER": (40, 40, 230), "CAUTION": (0, 200, 255), "SAFE": (80, 200, 60)}  # BGR


WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


@contextlib.contextmanager
def _in_weights_dir():
    """Ultralytics downloads missing assets into the working directory; keep them in weights/."""
    cwd = os.getcwd()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.chdir(WEIGHTS_DIR)
    try:
        yield
    finally:
        os.chdir(cwd)


class Detector:
    """Ultralytics YOLO segmentation model (COCO classes) or YOLOE with open-vocabulary prompts."""

    # support surfaces, signals and carried items are not obstacles on their own
    IGNORE = ("dining table", "traffic light", "handbag", "backpack", "tie", "umbrella")

    def __init__(self, weights, conf=0.3, device="cpu", prompts=None, imgsz=1024, ignore=IGNORE):
        from ultralytics import YOLO
        path = weights if os.path.isabs(weights) else os.path.join(WEIGHTS_DIR, weights)
        with _in_weights_dir():
            self.model = YOLO(path if os.path.exists(path) else weights)
            if prompts:
                names = [p.strip() for p in prompts if p.strip()]
                self.model.set_classes(names, self.model.get_text_pe(names))
        self.conf = conf
        self.device = device
        self.imgsz = imgsz
        self.ignore = set(ignore or ())

    def _parse(self, r, shape):
        dets = []
        if r.boxes is None or len(r.boxes) == 0 or r.masks is None:
            return dets
        h, w = shape[:2]
        masks = r.masks.data.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(r.boxes)
        for i in range(len(r.boxes)):
            name = r.names[int(r.boxes.cls[i])]
            if name in self.ignore:
                continue
            m = masks[i]
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            dets.append(Detection(name, float(r.boxes.conf[i]),
                                  tuple(float(v) for v in r.boxes.xyxy[i].tolist()), m > 0.5, int(ids[i])))
        return dets

    def detect(self, img):
        r = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, retina_masks=True,
                               device=self.device, verbose=False)[0]
        return self._parse(r, img.shape)

    def track(self, img):
        r = self.model.track(img, conf=self.conf, imgsz=self.imgsz, retina_masks=True, persist=True,
                             tracker="bytetrack.yaml", device=self.device, verbose=False)[0]
        return self._parse(r, img.shape)


def classify(dist, in_path, profile: Profile, ttc=float("inf")):
    if in_path and (dist < profile.danger_m or ttc < profile.ttc_danger_s):
        return "DANGER"
    if (in_path and (dist < profile.caution_m or ttc < profile.ttc_caution_s)) or dist < profile.danger_m:
        return "CAUTION"
    return "SAFE"


def measure_objects(dets, depth, f, cx, cy, profile: Profile, reliable=None, plane=None,
                    near_pct=20, min_px=30):
    """Distance, lateral position, size and warning level of every detected object."""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    out = []
    kernel = np.ones((5, 5), np.uint8)
    frame = ground_frame(plane) if plane is not None else None
    for det in dets:
        core = cv2.erode(det.mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        if core.sum() < min_px:
            core = det.mask
        m = core & np.isfinite(depth) & (depth > 0)
        if m.sum() < min_px:
            continue
        z = depth[m]
        x = (u[m] - cx) * z / f
        y = (v[m] - cy) * z / f
        info = ObjectInfo(det, float(np.percentile(z, near_pct)), float(np.median(z)),
                          float(np.median(x)), float(np.percentile(x, 5)), float(np.percentile(x, 95)))
        if reliable is not None:
            info.valid_ratio = float((reliable & core).sum() / max(1, core.sum()))
        if frame is not None:
            right, fwd, up = frame
            P = np.stack([x, y, z], 1)
            hg = P @ up + plane.d
            xg = P @ right
            info.height_m = float(np.percentile(hg, 95))
            info.width_m = float(np.percentile(xg, 95) - np.percentile(xg, 5))
            info.x_lo, info.x_hi = float(np.percentile(xg, 5)), float(np.percentile(xg, 95))
            info.x_m = float(np.median(xg))
        info.in_path = info.x_hi > -profile.half_width_m and info.x_lo < profile.half_width_m
        info.level = classify(info.dist_m, info.in_path, profile)
        out.append(info)
    out.sort(key=lambda o: o.dist_m)
    return out


DEFAULT_PROMPTS = ["person", "bicycle", "car", "chair", "table", "bottle", "cup", "book", "monitor", "laptop",
                   "bag", "box", "door", "stairs", "pole", "trash can", "plant", "cable", "lamp"]


class GroundedSAM2:
    """Open-vocabulary objects: Grounding DINO boxes from text prompts, then SAM 2.1 masks.

    Both are Apache-2.0 models loaded through Hugging Face transformers."""

    def __init__(self, prompts, box_threshold=0.30, text_threshold=0.25, device="cuda",
                 gdino="IDEA-Research/grounding-dino-base", sam="facebook/sam2.1-hiera-large"):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        cache = os.path.join(WEIGHTS_DIR, "hf")
        self.torch = torch
        self.device = device
        self.prompts = [p.strip() for p in prompts if p.strip()]
        self.box_t, self.text_t = box_threshold, text_threshold
        self.gp = AutoProcessor.from_pretrained(gdino, cache_dir=cache)
        self.gd = AutoModelForZeroShotObjectDetection.from_pretrained(gdino, cache_dir=cache).to(device).eval()
        self.sp = Sam2Processor.from_pretrained(sam, cache_dir=cache)
        self.sm = Sam2Model.from_pretrained(sam, cache_dir=cache).to(device).eval()

    def boxes(self, img):
        torch = self.torch
        h, w = img.shape[:2]
        text = ". ".join(self.prompts) + "."
        inputs = self.gp(images=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), text=text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.gd(**inputs)
        r = self.gp.post_process_grounded_object_detection(out, inputs.input_ids, threshold=self.box_t,
                                                           text_threshold=self.text_t, target_sizes=[(h, w)])[0]
        labels = r.get("text_labels", r.get("labels"))
        return [(str(l) or "object", float(s), [float(v) for v in b]) for l, s, b in
                zip(labels, r["scores"].cpu(), r["boxes"].cpu())]

    def masks(self, img, boxes):
        torch = self.torch
        if not boxes:
            return []
        inputs = self.sp(images=cv2.cvtColor(img, cv2.COLOR_BGR2RGB), input_boxes=[[list(b) for b in boxes]],
                         return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.sm(**inputs, multimask_output=False)
        m = self.sp.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        return [mm[0].numpy() > 0.5 for mm in m]

    # support surfaces are where things stand, not obstacles
    IGNORE = set(Detector.IGNORE) | {"table", "desk", "floor", "ground", "road", "sidewalk", "wall", "ceiling"}

    def detect(self, img):
        found = [x for x in self.boxes(img) if x[0] not in self.IGNORE]
        # one box per object: drop boxes that mostly cover a higher-scoring box of the same name
        found.sort(key=lambda x: -x[1])
        keep = []
        for name, score, b in found:
            if all(_iou(b, k[2]) < 0.6 for k in keep):
                keep.append((name, score, b))
        masks = self.masks(img, [k[2] for k in keep])
        return [Detection(n, s, tuple(b), m) for (n, s, b), m in zip(keep, masks) if m.sum() > 50]


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


# Typical real sizes (m) of everyday objects: (measured dimension, low, high).  Used only as a sanity
# check of the metric scale when the slide length was estimated instead of measured.
SIZE_PRIORS = {
    "bottle": ("height", 0.15, 0.40), "cup": ("height", 0.07, 0.16), "wine glass": ("height", 0.15, 0.30),
    "tv": ("width", 0.45, 1.70), "laptop": ("width", 0.28, 0.42), "keyboard": ("width", 0.30, 0.50),
    "chair": ("height", 0.70, 1.20), "car": ("height", 1.30, 2.00), "person": ("height", 1.40, 2.00),
}


def size_check(objs, image_shape, margin=3):
    """Objects whose estimated size is far outside the usual range (ignores objects cut by the border)."""
    h, w = image_shape[:2]
    notes = []
    for o in objs:
        prior = SIZE_PRIORS.get(o.det.name)
        if prior is None:
            continue
        ys, xs = np.nonzero(o.det.mask)
        if xs.min() < margin or ys.min() < margin or xs.max() >= w - margin or ys.max() >= h - margin:
            continue
        dim, lo, hi = prior
        val = o.height_m if dim == "height" else o.width_m
        if not np.isfinite(val):
            continue
        if val < lo / 1.3 or val > hi * 1.3:
            notes.append(f"{o.det.name} {dim} {val:.2f} m is outside the usual {lo:.2f}-{hi:.2f} m: "
                         "the estimated scale is probably off, measure the slide length")
    return notes
