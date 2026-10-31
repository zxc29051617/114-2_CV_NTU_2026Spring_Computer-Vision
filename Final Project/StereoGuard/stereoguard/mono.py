"""Monocular metric depth: MoGe-2 (metric point map + field of view) or Depth Anything V2 Metric."""
import os

import cv2
import numpy as np

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE = os.path.join(APP_ROOT, "weights", "hf")


class MoGe2:
    """MoGe-2 (Wang et al. 2025).  Returns metric depth (m) and the horizontal FoV (deg)."""

    name = "MoGe-2"

    def __init__(self, repo="Ruicheng/moge-2-vits-normal", device="cpu"):
        import torch
        from moge.model.v2 import MoGeModel
        self.torch = torch
        self.device = device
        self.model = MoGeModel.from_pretrained(repo, cache_dir=HF_CACHE).to(device).eval()

    def __call__(self, img_bgr, hfov_deg=None, max_side=1024):
        torch = self.torch
        h, w = img_bgr.shape[:2]
        s = min(1.0, max_side / max(h, w))
        img = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img_bgr
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).float().div(255).permute(2, 0, 1)
        with torch.inference_mode():
            out = self.model.infer(x.to(self.device), fov_x=hfov_deg, use_fp16=self.device != "cpu")
        depth = out["depth"].float().cpu().numpy()
        mask = out["mask"].cpu().numpy().astype(bool) if "mask" in out else np.isfinite(depth)
        depth[~mask] = np.nan
        fx_norm = float(out["intrinsics"][0, 0].cpu())
        fov = float(np.degrees(2 * np.arctan(0.5 / fx_norm)))
        if s < 1:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        return depth.astype(np.float32), fov


class DepthAnythingV2Metric:
    """Depth Anything V2 metric head (indoor = Hypersim, outdoor = Virtual KITTI)."""

    name = "Depth Anything V2 Metric"

    def __init__(self, variant="Indoor", size="Small", device="cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        repo = f"depth-anything/Depth-Anything-V2-Metric-{variant}-{size}-hf"
        self.torch = torch
        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(repo, cache_dir=HF_CACHE)
        self.model = AutoModelForDepthEstimation.from_pretrained(repo, cache_dir=HF_CACHE).to(device).eval()

    def __call__(self, img_bgr, hfov_deg=None):
        torch = self.torch
        h, w = img_bgr.shape[:2]
        inputs = self.proc(images=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            pred = self.model(**inputs).predicted_depth
        depth = torch.nn.functional.interpolate(pred[:, None], size=(h, w), mode="bicubic",
                                                align_corners=False)[0, 0].cpu().numpy()
        return depth.astype(np.float32), None


class DepthPro:
    """Apple Depth Pro (ICLR 2025): metric depth and focal length from one image (fp16 on the GPU)."""

    name = "Depth Pro"

    def __init__(self, device="cuda"):
        import torch
        from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
        self.torch = torch
        self.device = device
        dtype = torch.float16 if device != "cpu" else torch.float32
        self.proc = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf", cache_dir=HF_CACHE)
        self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf", cache_dir=HF_CACHE,
                                                                torch_dtype=dtype).to(device).eval()
        self.dtype = dtype

    def __call__(self, img_bgr, hfov_deg=None):
        torch = self.torch
        h, w = img_bgr.shape[:2]
        inputs = self.proc(images=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), return_tensors="pt").to(self.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        with torch.inference_mode():
            out = self.model(**inputs)
        post = self.proc.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]
        depth = post["predicted_depth"].float().cpu().numpy()
        f_pred = float(post["focal_length"])
        fov = float(np.degrees(2 * np.arctan(w / (2 * f_pred))))
        if hfov_deg:                      # metric depth scales with the focal length
            f_true = w / (2 * np.tan(np.radians(hfov_deg) / 2))
            depth = depth * f_true / f_pred
        return depth.astype(np.float32), fov


class UniDepthV2:
    """UniDepthV2 (2025): metric depth + camera, optionally conditioned on known intrinsics."""

    name = "UniDepthV2"

    def __init__(self, repo="lpiccinelli/unidepth-v2-vitl14", device="cuda"):
        import sys
        import torch
        import types
        code = os.path.join(APP_ROOT, "third_party", "UniDepth")
        if code not in sys.path:
            sys.path.insert(0, code)
        sys.modules.setdefault("wandb", types.ModuleType("wandb"))   # only used for training logs
        from unidepth.models import UniDepthV2 as _U
        self.torch = torch
        self.device = device
        self.model = _U.from_pretrained(repo, cache_dir=HF_CACHE).to(device).eval()

    def __call__(self, img_bgr, hfov_deg=None):
        torch = self.torch
        h, w = img_bgr.shape[:2]
        rgb = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).to(self.device)
        K = None
        if hfov_deg:
            f = w / (2 * np.tan(np.radians(hfov_deg) / 2))
            K = torch.tensor([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]], dtype=torch.float32,
                             device=self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device != "cpu"):
            pred = self.model.infer(rgb, K) if K is not None else self.model.infer(rgb)
        depth = pred["depth"][0, 0].float().cpu().numpy()
        f_pred = float(pred["intrinsics"][0, 0, 0])
        fov = float(np.degrees(2 * np.arctan(w / (2 * f_pred))))
        return depth.astype(np.float32), fov


class VideoDepthAnythingMetric:
    """Metric Video Depth Anything (CVPR 2025): temporally consistent metric depth for a whole video.

    Small = Apache-2.0, Large = CC-BY-NC-4.0 (weights on Hugging Face, code in third_party/)."""

    name = "Video Depth Anything (metric)"
    CONFIGS = {"vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
               "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}}
    REPOS = {"vits": ("depth-anything/Metric-Video-Depth-Anything-Small", "metric_video_depth_anything_vits.pth"),
             "vitl": ("depth-anything/Metric-Video-Depth-Anything-Large", "metric_video_depth_anything_vitl.pth")}

    def __init__(self, encoder="vitl", device="cuda"):
        import sys
        import torch
        from huggingface_hub import hf_hub_download
        code = os.path.join(APP_ROOT, "third_party", "Video-Depth-Anything")
        if code not in sys.path:
            sys.path.insert(0, code)
        from video_depth_anything.video_depth import VideoDepthAnything
        repo, fname = self.REPOS[encoder]
        state = torch.load(hf_hub_download(repo, fname, cache_dir=HF_CACHE), map_location="cpu", weights_only=True)
        self.model = VideoDepthAnything(**self.CONFIGS[encoder], metric=True)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(device).eval()
        self.device = device

    def __call__(self, frames_rgb, fps=10):
        """frames_rgb: (T, H, W, 3) uint8 RGB -> (T, H, W) depth in meters."""
        depths, _ = self.model.infer_video_depth(frames_rgb, fps, input_size=518, device=self.device, fp32=False)
        return np.stack(depths).astype(np.float32)


class Metric3Dv2:
    """Metric3D v2 ViT-L (TPAMI 2024): metric depth in a canonical camera, rescaled by the focal length."""

    name = "Metric3D v2"

    def __init__(self, device="cuda"):
        import sys
        import torch
        from huggingface_hub import hf_hub_download
        code = os.path.join(APP_ROOT, "third_party", "Metric3D")
        if code not in sys.path:
            sys.path.insert(0, code)
        import types
        from mmengine import Config
        if "mmcv" not in sys.modules:                 # mono.utils.comm only needs mmcv for env logging
            mmcv = types.ModuleType("mmcv")
            mmcv.utils = types.ModuleType("mmcv.utils")
            mmcv.utils.collect_env = lambda: {}
            sys.modules["mmcv"], sys.modules["mmcv.utils"] = mmcv, mmcv.utils
        from mono.model.monodepth_model import get_configured_monodepth_model
        cfg = Config.fromfile(os.path.join(code, "mono/configs/HourglassDecoder/vit.raft5.large.py"))
        self.model = get_configured_monodepth_model(cfg)
        ckpt = hf_hub_download("JUGGHM/Metric3D", "metric_depth_vit_large_800k.pth", cache_dir=HF_CACHE)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"],
                                   strict=False)
        self.model = self.model.to(device).eval()
        self.torch, self.device = torch, device

    def __call__(self, img_bgr, hfov_deg=None):
        torch = self.torch
        h, w = img_bgr.shape[:2]
        f = w / (2 * np.tan(np.radians(hfov_deg or 60.0) / 2))
        ih, iw = 616, 1064                                   # input size of the ViT models
        s = min(ih / h, iw / w)
        rgb = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_LINEAR)
        mean = np.array([123.675, 116.28, 103.53], np.float32)
        std = np.array([58.395, 57.12, 57.375], np.float32)
        ph, pw = ih - rgb.shape[0], iw - rgb.shape[1]
        top, left = ph // 2, pw // 2
        pad = cv2.copyMakeBorder(rgb, top, ph - top, left, pw - left, cv2.BORDER_CONSTANT, value=mean.tolist())
        x = torch.from_numpy(((pad - mean) / std).transpose(2, 0, 1)).float()[None].to(self.device)
        with torch.inference_mode():
            depth, _, _ = self.model.inference({"input": x})
        d = depth[0, 0, top:top + rgb.shape[0], left:left + rgb.shape[1]].float().cpu().numpy()
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) * (f * s / 1000.0)   # canonical f = 1000
        return np.clip(d, 0, 300).astype(np.float32), hfov_deg
