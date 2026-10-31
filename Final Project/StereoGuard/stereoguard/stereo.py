"""Stereo matchers: Fast-FoundationStereo (pretrained, ONNX) and the Team 18 Census matcher."""
import importlib.util
import os
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP_ROOT = os.path.dirname(HERE)
TEAM18_BONUS = os.path.join(os.path.dirname(APP_ROOT), "Stereo Matching Team_Team18", "Stereo Matching_bonus")
DEFAULT_FFS = os.path.join(APP_ROOT, "weights", "c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx")

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], np.float32)


def gpu_free_gb():
    """Free GPU memory in GB (0 if there is no NVIDIA GPU)."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return 0.0
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split()
        return float(out[0]) / 1024 if out else 0.0
    except Exception:
        return 0.0


def resolve_device(device, need_gb=5.0):
    """'auto' -> 'cuda' only when the GPU has enough free memory (the GPU may be shared)."""
    if device != "auto":
        return device
    return "cuda" if gpu_free_gb() >= need_gb else "cpu"


def _providers(device):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if device == "cuda" and "CUDAExecutionProvider" in avail:
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # use the CUDA / cuDNN libraries installed with PyTorch
        opts = {"gpu_mem_limit": int(3.6 * 1024 ** 3), "arena_extend_strategy": "kSameAsRequested",
                "cudnn_conv_algo_search": "HEURISTIC"}
        return [("CUDAExecutionProvider", opts), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class FastFoundationStereo:
    """C-Fast-FoundationStereo (NVIDIA, CVPR 2026) exported to ONNX with dynamic input size.

    Input: rectified left/right BGR images.  Output: left disparity in pixels (float32).
    """

    name = "Fast-FoundationStereo"
    range_limit = 180  # the exported cost volume covers 192 px; stay a little below it

    def __init__(self, onnx_path=DEFAULT_FFS, device="cpu", threads=None):
        import onnxruntime as ort
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"{onnx_path} not found: run  python scripts/download_weights.py")
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=_providers(device))
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.output = self.sess.get_outputs()[0].name

    @staticmethod
    def _prep(img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        return x.transpose(2, 0, 1)[None]

    def _run(self, left, right):
        h, w = left.shape[:2]
        ph, pw = (-h) % 32, (-w) % 32
        if ph or pw:
            left = cv2.copyMakeBorder(left, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
            right = cv2.copyMakeBorder(right, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
        out = self.sess.run([self.output], {self.inputs[0]: self._prep(left), self.inputs[1]: self._prep(right)})[0]
        return out.reshape(out.shape[-2:])[:h, :w].astype(np.float32)

    def __call__(self, left, right, with_right=False):
        disp = self._run(left, right)
        if not with_right:
            return disp, None
        # right-view disparity from the mirrored pair (used for the left-right check)
        disp_r = self._run(cv2.flip(right, 1), cv2.flip(left, 1))[:, ::-1].copy()
        return disp, disp_r


class FoundationStereoTorch:
    """FoundationStereo (NVIDIA, CVPR 2025), the official PyTorch model with the ViT-L checkpoint.

    Zero-shot state of the art on Middlebury / ETH3D; heavier than Fast-FoundationStereo.
    Runs in fp16 autocast on the GPU.  Code from github.com/NVlabs/FoundationStereo (non-commercial
    NVIDIA license), placed in third_party/ by scripts/download_weights.py.
    """

    name = "FoundationStereo"
    range_limit = 400  # cfg max_disp = 416

    def __init__(self, ckpt=os.path.join(APP_ROOT, "weights", "foundationstereo", "23-51-11", "model_best_bp2.pth"),
                 device="cuda", valid_iters=32, hierarchical_above=1200):
        import sys
        import torch
        from omegaconf import OmegaConf
        code = os.path.join(APP_ROOT, "third_party", "FoundationStereo")
        if not os.path.isdir(code) or not os.path.exists(ckpt):
            raise FileNotFoundError("FoundationStereo code / weights missing: run  python scripts/download_weights.py "
                                    "--only foundationstereo")
        if code not in sys.path:
            sys.path.insert(0, code)
        os.environ.setdefault("XFORMERS_DISABLED", "1")
        import types
        # Utils.py imports open3d only for point-cloud helpers that are not used here
        sys.modules.setdefault("open3d", types.ModuleType("open3d"))
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt), "cfg.yaml"))
        cfg.setdefault("vit_size", "vitl")
        cfg.valid_iters = valid_iters
        self.torch, self.Padder = torch, InputPadder
        # build the network on the GPU and memory-map the 3.3 GB checkpoint, so that host RAM stays low
        with torch.device(device):
            self.model = FoundationStereo(cfg)
        state = torch.load(ckpt, map_location=device, mmap=True, weights_only=False)["model"]
        self.model.load_state_dict(state)
        del state
        self.device = device
        self.model.to(device).eval()
        self.iters = valid_iters
        self.hier = hierarchical_above

    def _run(self, left, right):
        torch = self.torch
        a = torch.as_tensor(cv2.cvtColor(left, cv2.COLOR_BGR2RGB)).to(self.device).float()[None].permute(0, 3, 1, 2)
        b = torch.as_tensor(cv2.cvtColor(right, cv2.COLOR_BGR2RGB)).to(self.device).float()[None].permute(0, 3, 1, 2)
        padder = self.Padder(a.shape, divis_by=32, force_square=False)
        a, b = padder.pad(a, b)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device != "cpu"):
            if max(left.shape[:2]) > self.hier:
                d = self.model.run_hierachical(a, b, iters=self.iters, test_mode=True, small_ratio=0.5)
            else:
                d = self.model.forward(a, b, iters=self.iters, test_mode=True)
        return padder.unpad(d.float()).cpu().numpy().reshape(left.shape[:2]).astype(np.float32)

    def __call__(self, left, right, with_right=False):
        disp = self._run(left, right)
        if not with_right:
            return disp, None
        disp_r = self._run(cv2.flip(right, 1), cv2.flip(left, 1))[:, ::-1].copy()
        return disp, disp_r


def shift_right(right, d_off):
    """Shift the right view by d_off pixels so that the network only sees disparities d - d_off.

    Equivalent to moving the principal point (Middlebury's 'doffs'); used when the phone slide is
    long compared to the scene depth and disparities would exceed the matcher's search range.
    """
    if d_off <= 0:
        return right
    pad = cv2.copyMakeBorder(right, 0, 0, d_off, 0, cv2.BORDER_REPLICATE)
    return pad[:, :right.shape[1]]


def load_team18_module():
    """Import the original Team 18 computeDisp.py (Census + guided filter + WTA + refinement) unchanged."""
    import sys
    path = os.path.join(TEAM18_BONUS, "computeDisp.py")
    spec = importlib.util.spec_from_file_location("team18_computeDisp", path)
    mod = importlib.util.module_from_spec(spec)
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True          # leave the original Team 18 folder untouched
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = dont_write
    return mod


class Team18Census:
    """The original classical matcher; disparities are integers in [0, max_disp]."""

    name = "Team18 Census"
    range_limit = 128  # cost volume memory grows with H x W x range

    def __init__(self, max_disp=96):
        self.mod = load_team18_module()
        self.max_disp = int(max_disp)

    def __call__(self, left, right, with_right=False):
        final, reliable = self.mod.computeDispWithConfidence(left, right, self.max_disp, confidence_margin=1.0)
        return final.astype(np.float32), reliable.astype(np.float32)


def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t
