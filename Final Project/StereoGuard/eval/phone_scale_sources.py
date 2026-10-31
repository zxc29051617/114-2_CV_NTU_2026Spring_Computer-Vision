"""Team 18 photos: how long was the slide?  Every scale source, and the object sizes it implies.

    python eval/phone_scale_sources.py

The photos have no EXIF and the slide was not measured.  The geometry from FoundationStereo is
fixed; only the global scale (slide length B) changes, and every distance / size scales with B.
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from stereoguard import fusion, pipeline  # noqa: E402

SCENE = os.path.join(os.path.dirname(ROOT), "Stereo Matching Team_Team18", "Stereo Matching_bonus", "phone_data",
                     "scene1")
SOURCES = ["dav2s", "dav2l", "moge2", "depthpro", "unidepth"]
LABELS = {"dav2s": "Depth Anything V2 Metric-S", "dav2l": "Depth Anything V2 Metric-L", "moge2": "MoGe-2 ViT-S",
          "depthpro": "Depth Pro", "unidepth": "UniDepthV2-L", "mapanything": "MapAnything (camera poses)"}


def main():
    img1 = cv2.imread(os.path.join(SCENE, "left.jpg"))
    img2 = cv2.imread(os.path.join(SCENE, "right.jpg"))
    models = pipeline.Models(device="auto", scale="none")
    # geometry for a reference slide of 9 cm (plane fit and size thresholds are metric); every length
    # is then rescaled linearly to the slide length implied by each source
    B_REF = 0.09
    res = pipeline.analyse_phone_pair(img1, img2, models, baseline_m=B_REF, path1=os.path.join(SCENE, "left.jpg"))
    w = res.image.shape[1]
    hfov = float(np.degrees(2 * np.arctan(w / (2 * res.f))))
    unit = {o.det.name: o for o in res.objects}
    rows = []
    for name in SOURCES:
        z, _ = models.get(name)(res.image, hfov_deg=hfov)
        models.release(name)
        b, _ = fusion.estimate_baseline(res.disp, res.f, z, res.confident)
        rows.append((name, b))
    ma = models.get("mapanything")
    s = 1600 / img1.shape[1]
    mv = ma([cv2.resize(img1, None, fx=s, fy=s, interpolation=cv2.INTER_AREA),
             cv2.resize(img2, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)])
    models.release("mapanything")
    rows.append(("mapanything", mv.baseline()))
    table = []
    for name, b in rows:
        r = {"source": LABELS[name], "baseline_cm": round(100 * b, 1)}
        for k, attr in [("bottle", "dist_m"), ("bottle", "height_m"), ("bottle", "width_m"), ("tv", "width_m")]:
            if k in unit:
                r[f"{k}_{attr}"] = round(getattr(unit[k], attr) * b / B_REF, 3)
        table.append(r)
        print(r, flush=True)
    out = os.path.join(ROOT, "results", "phone")
    os.makedirs(out, exist_ok=True)
    json.dump({"focal": res.intrinsics_source, "rows": table}, open(os.path.join(out, "scale_sources.json"), "w"),
              indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
