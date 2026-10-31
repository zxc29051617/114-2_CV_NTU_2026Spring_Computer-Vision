"""Camera intrinsics for phone photos.

Focal length priority:
  1. a value given by the user (35 mm equivalent, mm)
  2. EXIF FocalLengthIn35mmFilm (or FocalLength + sensor crop factor)
  3. the field of view predicted from the photos (VGGT on a GPU, MoGe-2 on a CPU)
  4. a typical phone main camera (26 mm equivalent)

A sideways phone slide is (almost) a pure translation, which is a degenerate case for
self-calibration from two views, so the focal length cannot be recovered from the image
pair itself and has to come from one of the sources above.
"""
from dataclasses import dataclass

import numpy as np

FULL_FRAME_DIAG_MM = 43.267  # diagonal of a 36 x 24 mm frame
DEFAULT_PHONE_F35 = 26.0     # main (1x) camera of most recent phones


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    source: str = "unknown"

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)

    def scaled(self, s):
        return Intrinsics(self.fx * s, self.fy * s, (self.cx + 0.5) * s - 0.5, (self.cy + 0.5) * s - 0.5,
                          int(round(self.width * s)), int(round(self.height * s)), self.source)

    @property
    def hfov_deg(self):
        return float(np.degrees(2 * np.arctan(self.width / (2 * self.fx))))


def f35_to_pixels(f35_mm, width, height):
    """35 mm equivalent focal length -> focal length in pixels (matches the image diagonal)."""
    return f35_mm * np.hypot(width, height) / FULL_FRAME_DIAG_MM


def from_f35(f35_mm, width, height, source):
    f = f35_to_pixels(f35_mm, width, height)
    return Intrinsics(f, f, (width - 1) / 2, (height - 1) / 2, width, height, source)


def from_hfov(hfov_deg, width, height, source):
    f = width / (2 * np.tan(np.radians(hfov_deg) / 2))
    return Intrinsics(f, f, (width - 1) / 2, (height - 1) / 2, width, height, source)


def read_exif_f35(path):
    """Return the 35 mm equivalent focal length stored in the photo, or None."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            exif = im.getexif()
            sub = exif.get_ifd(0x8769) if exif else {}
            f35 = sub.get(0xA405) or exif.get(0xA405)  # FocalLengthIn35mmFilm
            if f35:
                return float(f35)
    except Exception:
        pass
    return None


def guess_intrinsics(path, width, height, f35_user=None, hfov_model=None, model_name="model"):
    """Pick the best available focal length source (see module docstring)."""
    if f35_user:
        return from_f35(float(f35_user), width, height, f"user ({f35_user:g} mm eq.)")
    f35 = read_exif_f35(path) if path else None
    if f35:
        return from_f35(f35, width, height, f"EXIF ({f35:g} mm eq.)")
    if hfov_model:
        return from_hfov(hfov_model, width, height, f"{model_name} FoV ({hfov_model:.1f} deg)")
    return from_f35(DEFAULT_PHONE_F35, width, height, f"phone default ({DEFAULT_PHONE_F35:g} mm eq.)")
