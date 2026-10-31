"""Download model weights into weights/ and third-party model code into third_party/ (both ignored by git).

    python scripts/download_weights.py                 # what the demos need: ffs fs yolo dav2 moge2
    python scripts/download_weights.py --only fs       # one model
    python scripts/download_weights.py --all           # also every model used in eval/

Model                     weights from                                   license
ffs   Fast-FoundationStereo  HF nvidia/c-fast-foundationstereo (ONNX)      NVIDIA Open Model Agreement
fs    FoundationStereo ViT-L Google Drive of NVlabs/FoundationStereo       NVIDIA, non-commercial
yolo  YOLO11-seg             GitHub ultralytics/assets                     AGPL-3.0
yoloe YOLOE-26-seg           GitHub ultralytics/assets                     AGPL-3.0
gsam  Grounding DINO + SAM2.1 HF IDEA-Research / facebook                  Apache-2.0
dav2  Depth Anything V2 Metric Indoor S / L  HF depth-anything             Apache-2.0 / CC-BY-NC-4.0
moge2 MoGe-2 ViT-S           HF Ruicheng/moge-2-vits-normal                MIT (code)
depthpro Depth Pro           HF apple/DepthPro-hf                          Apple ML research license
unidepth UniDepthV2-L        HF lpiccinelli/unidepth-v2-vitl14             CC-BY-NC-4.0
mapanything MapAnything      HF facebook/map-anything-apache               Apache-2.0
vggt  VGGT-1B                HF facebook/VGGT-1B                           non-commercial (VGGT license)
vda   Metric Video Depth Anything S / L  HF depth-anything                 Apache-2.0 / CC-BY-NC-4.0
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "weights")
HF = os.path.join(WEIGHTS, "hf")
THIRD = os.path.join(ROOT, "third_party")
UL = "https://github.com/ultralytics/assets/releases/download/v8.3.0"

CODE = {  # repository, pinned commit
    "FoundationStereo": ("https://github.com/NVlabs/FoundationStereo.git", "6e8806816b533e4d13ddbb95ffa907b797060a62"),
    "map-anything": ("https://github.com/facebookresearch/map-anything.git", "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"),
    "UniDepth": ("https://github.com/lpiccinelli-eth/UniDepth.git", "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"),
    "Video-Depth-Anything": ("https://github.com/DepthAnything/Video-Depth-Anything.git",
                             "4f5ae23172ba60fd7bc11ef671cca678842c7072"),
}


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"  ok  {os.path.relpath(dst, ROOT)}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"  get {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(dst + ".part", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(dst + ".part", dst)


def clone(name):
    url, commit = CODE[name]
    dst = os.path.join(THIRD, name)
    if os.path.isdir(dst):
        print(f"  ok  third_party/{name}")
        return
    os.makedirs(THIRD, exist_ok=True)
    subprocess.run(["git", "clone", "-q", url, dst], check=True)
    subprocess.run(["git", "-C", dst, "checkout", "-q", commit], check=True)
    print(f"  ok  third_party/{name} @ {commit[:8]}")


def hf(repo, files=None):
    from huggingface_hub import hf_hub_download, snapshot_download
    if files:
        for f in files:
            hf_hub_download(repo, f, cache_dir=HF)
    else:
        snapshot_download(repo, cache_dir=HF, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.pt", "*.pth"])
    print(f"  ok  {repo}")


def get_ffs():
    from huggingface_hub import hf_hub_download
    name = "onnx/c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx"
    p = hf_hub_download("nvidia/c-fast-foundationstereo", name, cache_dir=HF)
    dst = os.path.join(WEIGHTS, os.path.basename(name))
    if not os.path.exists(dst):
        shutil.copyfile(p, dst)
    print(f"  ok  {os.path.relpath(dst, ROOT)}")


def get_fs():
    clone("FoundationStereo")
    dst = os.path.join(WEIGHTS, "foundationstereo")
    if os.path.exists(os.path.join(dst, "23-51-11", "model_best_bp2.pth")):
        print("  ok  weights/foundationstereo/23-51-11")
        return
    import gdown  # the official checkpoints are shared from Google Drive
    gdown.download_folder("https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf",
                          output=dst, quiet=True)
    print("  ok  weights/foundationstereo")


def get_yolo():
    fetch(f"{UL}/yolo11s-seg.pt", os.path.join(WEIGHTS, "yolo11s-seg.pt"))


def get_yoloe():
    for n in ["yoloe-26s-seg.pt", "mobileclip2_b.ts"]:
        fetch(f"{UL}/{n}", os.path.join(WEIGHTS, n))


def get_gsam():
    hf("IDEA-Research/grounding-dino-base")
    hf("facebook/sam2.1-hiera-large")


def get_dav2():
    hf("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")


def get_dav2l():
    hf("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")


def get_moge2():
    hf("Ruicheng/moge-2-vits-normal")


def get_depthpro():
    hf("apple/DepthPro-hf")


def get_unidepth():
    clone("UniDepth")
    hf("lpiccinelli/unidepth-v2-vitl14", ["config.json", "model.safetensors"])


def get_mapanything():
    clone("map-anything")
    hf("facebook/map-anything-apache", ["config.json", "model.safetensors"])


def get_vggt():
    hf("facebook/VGGT-1B", ["config.json", "model.safetensors"])


def get_vda():
    clone("Video-Depth-Anything")
    hf("depth-anything/Metric-Video-Depth-Anything-Small", ["metric_video_depth_anything_vits.pth"])
    hf("depth-anything/Metric-Video-Depth-Anything-Large", ["metric_video_depth_anything_vitl.pth"])


GETTERS = {"ffs": get_ffs, "fs": get_fs, "yolo": get_yolo, "yoloe": get_yoloe, "gsam": get_gsam,
           "dav2": get_dav2, "dav2l": get_dav2l, "moge2": get_moge2, "depthpro": get_depthpro,
           "unidepth": get_unidepth, "mapanything": get_mapanything, "vggt": get_vggt, "vda": get_vda}
DEFAULT = ["ffs", "fs", "yolo", "dav2", "moge2", "depthpro"]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", choices=list(GETTERS), default=None)
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    for k in a.only or (list(GETTERS) if a.all else DEFAULT):
        print(k)
        try:
            GETTERS[k]()
        except Exception as e:  # keep going so one failed mirror does not block the rest
            print(f"  failed: {e}", file=sys.stderr)
