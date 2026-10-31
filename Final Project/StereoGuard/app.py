"""StereoGuard web demo (Gradio).

    python app.py            # then open http://127.0.0.1:7860
"""
import os
import tempfile

import cv2
import gradio as gr

from stereoguard import freespace, pipeline, viz
from stereoguard.mesh import depth_to_glb
from stereoguard.report import summary_dict

ROOT = os.path.dirname(os.path.abspath(__file__))
TEAM18_SCENE = os.path.join(os.path.dirname(ROOT), "Stereo Matching Team_Team18", "Stereo Matching_bonus",
                            "phone_data", "scene1")
DEVICE = os.environ.get("STEREOGUARD_DEVICE", "auto")
_MODELS = {}


def get_models(stereo, detector, prompts):
    key = (stereo, detector, prompts)
    if key not in _MODELS:
        _MODELS.clear()  # keep at most one set of networks in memory
        det_w = {"coco": "yolo11s-seg.pt", "gsam": "gsam", "yoloe": "yoloe-26s-seg.pt"}[detector]
        plist = [p.strip() for p in prompts.split(",") if p.strip()] if detector != "coco" else None
        _MODELS[key] = pipeline.Models(device=DEVICE, stereo=stereo, detector=det_w, prompts=plist)
    return _MODELS[key]


def run_pair(left_path, right_path, baseline_cm, f35, profile, danger, caution, stereo, detector, prompts):
    if not left_path or not right_path:
        raise gr.Error("請上傳左、右兩張照片")
    models = get_models(stereo, detector, prompts or "")
    prof = freespace.PROFILES[profile]
    prof = freespace.Profile(**{**prof.__dict__, "danger_m": float(danger), "caution_m": float(caution)})
    img1, img2 = cv2.imread(left_path), cv2.imread(right_path)
    res = pipeline.analyse_phone_pair(img1, img2, models, baseline_m=baseline_cm / 100 if baseline_cm else None,
                                      f35=f35 or None, path1=left_path, profile=prof)
    s = summary_dict(res)
    ar = viz.draw_ar(res.image, res.objects, res.fs, res.profile)[..., ::-1]
    depth = viz.colorize_depth(res.depth)[..., ::-1]
    top = viz.draw_bev(res.fs, res.objects, res.profile)[..., ::-1] if res.fs is not None else None
    glb = os.path.join(tempfile.mkdtemp(), "scene.glb")
    depth_to_glb(res.depth, res.image, res.f, res.cx, res.cy, glb, max_depth=res.profile.range_m * 1.5)
    rows = [[o["name"], o["distance_m"], o["lateral_m"], o["height_m"], o["level"], "是" if o["in_path"] else "否",
             o["reliable_ratio"]] for o in s["objects"]]
    md = (f"**提示**：{s['hint']}\n\n"
          f"- 焦距：{s['focal_px']:.0f} px（{s['intrinsics']}）\n"
          f"- 移動距離（基線）：{s['baseline_cm']:.2f} cm（{'使用者輸入' if s['baseline_source'] == 'given' else '由 7 個單眼公尺深度模型估計，手機資料上平均誤差約 10%'}）\n"
          f"- 校正後垂直誤差中位數：{s['rectification']['median_abs_dy_px']:.2f} px，"
          f"可靠匹配比例：{100 * s['reliable_stereo_ratio']:.0f}%\n"
          f"- 耗時：" + "，".join(f"{k} {v:.1f}s" for k, v in s["timings_s"].items()))
    return ar, depth, top, glb, rows, md


def build():
    with gr.Blocks(title="StereoGuard") as demo:
        gr.Markdown("# StereoGuard：手機雙拍立體視覺避障\n"
                    "把手機水平移動拍兩張照片，系統會自動校正、估計深度（公尺）、辨識物體並給出警示與可走方向。")
        with gr.Tab("手機雙拍"):
            with gr.Row():
                left = gr.Image(type="filepath", label="左照片（先拍）")
                right = gr.Image(type="filepath", label="右照片（手機往右移後拍）")
            with gr.Row():
                baseline = gr.Number(value=None, label="移動距離 cm（不知道就留空，由模型估計）")
                f35 = gr.Number(value=None, label="等效焦距 mm（EXIF 沒有時可填，手機主鏡頭約 26）")
                profile = gr.Radio(list(freespace.PROFILES), value="tabletop", label="情境")
            with gr.Row():
                danger = gr.Slider(0.1, 20, value=0.4, step=0.05, label="危險距離 (m)")
                caution = gr.Slider(0.2, 40, value=0.8, step=0.05, label="注意距離 (m)")
            with gr.Row():
                stereo = gr.Radio([("自動（GPU 用 FoundationStereo）", "auto"), ("FoundationStereo", "fs"),
                                   ("Fast-FoundationStereo", "ffs"), ("第 18 組 Census", "census")], value="auto",
                                  label="立體匹配")
                detector = gr.Radio([("YOLO11-seg（COCO 80 類）", "coco"),
                                     ("Grounding DINO + SAM 2.1（開放詞彙）", "gsam"),
                                     ("YOLOE（開放詞彙）", "yoloe")], value="coco", label="物體偵測")
                prompts = gr.Textbox(value="bottle, book, monitor, cable, lamp, chair, person, door",
                                     label="開放詞彙物體名稱（逗號分隔）")
            go = gr.Button("分析", variant="primary")
            with gr.Row():
                ar = gr.Image(label="AR 標示")
                depth = gr.Image(label="深度圖")
            with gr.Row():
                top = gr.Image(label="俯視圖（障礙物與可走方向）")
                mesh = gr.Model3D(label="3D 場景", clear_color=(1, 1, 1, 1))
            table = gr.Dataframe(headers=["物體", "距離 m", "左右偏移 m", "高度 m", "等級", "在路徑上", "可靠比例"],
                                 label="物體列表")
            md = gr.Markdown()

            def _profile_defaults(p):
                pr = freespace.PROFILES[p]
                return gr.update(value=pr.danger_m), gr.update(value=pr.caution_m)
            profile.change(_profile_defaults, profile, [danger, caution])
            go.click(run_pair, [left, right, baseline, f35, profile, danger, caution, stereo, detector, prompts],
                     [ar, depth, top, mesh, table, md])
            if os.path.exists(os.path.join(TEAM18_SCENE, "left.jpg")):
                gr.Examples([[os.path.join(TEAM18_SCENE, "left.jpg"), os.path.join(TEAM18_SCENE, "right.jpg")]],
                            [left, right], label="第 18 組的手機照片")
    return demo


if __name__ == "__main__":
    build().launch(server_name=os.environ.get("HOST", "127.0.0.1"), server_port=int(os.environ.get("PORT", 7860)))
