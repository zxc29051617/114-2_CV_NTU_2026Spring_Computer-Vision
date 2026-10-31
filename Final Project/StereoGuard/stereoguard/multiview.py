"""Feed-forward multi-view 3D models: camera intrinsics, relative pose and dense depth from 2+ photos.

* MapAnything (Meta, 2025): metric scale, so the length of the phone slide comes out in meters.
  The Apache-2.0 checkpoint facebook/map-anything-apache is used by default.
* VGGT (Meta, CVPR 2025): cameras and depth up to an unknown scale.

Both run in bf16 on the GPU.  Inputs are BGR images of the same size; every model works at its own
resolution (multiple of 14 px, about 518 px wide) and the results are mapped back to the input size.
"""
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE = os.path.join(APP_ROOT, "weights", "hf")


@dataclass
class MultiViewResult:
    K: list               # per view 3x3 intrinsics in input pixels
    cam2world: list       # per view 4x4 OpenCV camera-to-world pose (metric for MapAnything)
    depth: list           # per view depth map at input resolution (NaN = masked)
    metric: bool

    def relative(self, i=0, j=1):
        """Pose of view j in the camera frame of view i: (R, t) with x_j = R x_i + t."""
        Ti, Tj = self.cam2world[i], self.cam2world[j]
        T = np.linalg.inv(Tj) @ Ti
        return T[:3, :3], T[:3, 3]

    def baseline(self, i=0, j=1):
        return float(np.linalg.norm(self.cam2world[j][:3, 3] - self.cam2world[i][:3, 3]))


def _load_on_device(cls, repo, device, patch_config=None):
    """Build the model directly on the GPU and load the safetensors there.

    from_pretrained() would first build an fp32 copy in CPU memory (and MapAnything would also fetch
    the DINOv2-giant ImageNet weights from torch hub), which does not fit next to other jobs."""
    import json
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    cfg = json.load(open(hf_hub_download(repo, "config.json", cache_dir=HF_CACHE)))
    if patch_config:
        patch_config(cfg)
    with torch.device(device):
        model = cls(**cfg)
    state = load_file(hf_hub_download(repo, "model.safetensors", cache_dir=HF_CACHE), device=device)
    missing, _ = model.load_state_dict(state, strict=False)
    # modules registered under two names (e.g. MapAnything's dpt_feature_head inside dense_head) are
    # stored once in the safetensors file; anything else missing is a real error
    names = {}
    for k, p in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False)):
        names.setdefault(id(p), []).append(k)
    tensors = dict(model.named_parameters(remove_duplicate=False))
    tensors.update(dict(model.named_buffers(remove_duplicate=False)))
    real = [k for k in missing if "num_batches_tracked" not in k
            and not any(a in state for a in names.get(id(tensors.get(k)), []))]
    if real:
        raise RuntimeError(f"{repo}: missing weights {real[:5]}")
    del state
    return model.to(device).eval()   # a few buffers are built with legacy CPU constructors


def _target_size(w, h, long_side=518, patch=14):
    if w >= h:
        tw = long_side
        th = max(patch, int(round(h * tw / w / patch)) * patch)
    else:
        th = long_side
        tw = max(patch, int(round(w * th / h / patch)) * patch)
    return tw, th


class MapAnythingModel:
    name = "MapAnything"
    metric = True

    def __init__(self, repo="facebook/map-anything-apache", device="cuda"):
        import torch
        code = os.path.join(APP_ROOT, "third_party", "map-anything")
        if code not in sys.path:
            sys.path.insert(0, code)
        from mapanything.models import MapAnything
        self.torch = torch
        self.device = device

        def no_hub_weights(cfg):
            cfg["encoder_config"]["torch_hub_pretrained"] = False
        self.model = _load_on_device(MapAnything, repo, device, no_hub_weights)

    def __call__(self, images):
        torch = self.torch
        from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT
        norm = IMAGE_NORMALIZATION_DICT["dinov2"]
        mean = torch.tensor(norm.mean).view(3, 1, 1)
        std = torch.tensor(norm.std).view(3, 1, 1)
        h, w = images[0].shape[:2]
        tw, th = _target_size(w, h)
        sx, sy = tw / w, th / h
        views = []
        for k, img in enumerate(images):
            rgb = cv2.cvtColor(cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            x = (torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0 - mean) / std
            views.append({"img": x[None].to(self.device), "true_shape": np.int32([[th, tw]]), "idx": k,
                          "instance": str(k), "data_norm_type": ["dinov2"]})
        with torch.inference_mode():
            preds = self.model.infer(views, memory_efficient_inference=False, use_amp=True, amp_dtype="bf16",
                                     apply_mask=True, mask_edges=True)
        Ks, poses, depths = [], [], []
        for p in preds:
            K = p["intrinsics"][0].float().cpu().numpy().copy()
            K[0] /= sx
            K[1] /= sy
            Ks.append(K)
            poses.append(p["camera_poses"][0].float().cpu().numpy())
            d = p["depth_z"][0, ..., 0].float().cpu().numpy()
            if "mask" in p:
                d = np.where(p["mask"][0, ..., 0].cpu().numpy() > 0, d, np.nan)
            depths.append(cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        return MultiViewResult(Ks, poses, depths, True)


class VGGTModel:
    name = "VGGT"
    metric = False

    def __init__(self, repo="facebook/VGGT-1B", device="cuda"):
        import torch
        from vggt.models.vggt import VGGT
        self.torch = torch
        self.device = device
        self.model = _load_on_device(VGGT, repo, device)

    def __call__(self, images):
        torch = self.torch
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        h, w = images[0].shape[:2]
        tw, th = _target_size(w, h)
        sx, sy = tw / w, th / h
        x = torch.stack([torch.from_numpy(cv2.cvtColor(cv2.resize(i, (tw, th), interpolation=cv2.INTER_AREA),
                                                       cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1) / 255.0
                         for i in images]).to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.model(x)
        extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], x.shape[-2:])
        Ks, poses, depths = [], [], []
        for k in range(len(images)):
            K = intr[0, k].float().cpu().numpy().copy()
            K[0] /= sx
            K[1] /= sy
            Ks.append(K)
            E = np.eye(4)
            E[:3] = extr[0, k].float().cpu().numpy()          # world-to-camera
            poses.append(np.linalg.inv(E))
            d = pred["depth"][0, k, ..., 0].float().cpu().numpy()
            depths.append(cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32))
        return MultiViewResult(Ks, poses, depths, False)
