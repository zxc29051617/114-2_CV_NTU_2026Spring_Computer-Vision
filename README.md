# NTU Computer Vision 2026 Spring

這個 repository 收錄臺大 114-2 電腦視覺（簡韶逸老師）課程的講義、三份作業與 Stereo Matching 期末專題。

內容從基礎影像處理與局部特徵開始，延伸到場景分類、CNN、projective geometry、panorama、stereo matching
與深度感知。

## 專案結構

```text
.
├── README.md
├── lectures/                    # 課程講義與 Structure from Motion 補充教材
├── HW1/                         # DoG 與 Joint Bilateral Filter
├── HW2/                         # 場景辨識與 CNN 影像分類
├── HW3/                         # Homography、AR、Panorama
└── Final Project/               # Stereo Matching 與深度感知應用
```

## HW1：DoG 與 Joint Bilateral Filter

題目說明：[`HW1/hw1_2026_spring.pdf`](HW1/hw1_2026_spring.pdf)  
說明文件：[`HW1/hw1_material/readme.md`](HW1/hw1_material/readme.md)

### Part 1：Difference of Gaussian

- 建立兩個 octave 的 Gaussian pyramid，相鄰影像相減得到 DoG pyramid。
- 在 3×3×3 鄰域中找局部極值作為 keypoints，並轉回原始影像尺度。

主要程式：`part1/DoG.py`

### Part 2：Joint Bilateral Filter

- 實作 bilateral filter 與 joint bilateral filter：spatial kernel 衡量距離，range kernel 衡量 guidance 影像的顏色差異。
- 比較不同 RGB 轉灰階權重產生的 guidance image，找出最能保留原始顏色差異的轉換。

主要程式：`part2/JBF.py`

```bash
cd HW1/hw1_material/part1 && python main.py --image_path ./testdata/1.png
cd ../part2 && python main.py --image_path ./testdata/1.png --setting_path ./testdata/1_setting.txt
```

## HW2：場景辨識與影像分類

題目說明：[`HW2/cv2026_hw2.pdf`](HW2/cv2026_hw2.pdf)  
環境與資料集：[`HW2/hw2/README.md`](HW2/hw2/README.md)

### Part 1：Bag-of-Words 場景辨識

- Tiny Image 特徵，以及 Dense SIFT + K-means visual words 的 Bag-of-SIFT 特徵。
- 自己實作 nearest-neighbor 分類器，輸出 accuracy 與 confusion matrix。

主要程式：`p1/p1.py`、`p1/utils.py`

```bash
cd HW2/hw2/p1
python p1.py --feature bag_of_sift --classifier nearest_neighbor --dataset_dir ../hw2_data/p1_data/
```

### Part 2：CNN 影像分類

- 自己設計的 CNN `MyNet`，以及調整後的 `ResNet18`。
- 訓練與驗證、learning curve、保存最佳 checkpoint，推論輸出 `filename,label` 的 CSV。

主要程式：`p2/p2_train.py`、`p2/model.py`、`p2/dataset.py`、`p2/config.py`、`p2/p2_inference.py`

## HW3：Projective Geometry

題目說明：[`HW3/hw3.pdf`](HW3/hw3.pdf)  
說明文件：[`HW3/hw3/README.md`](HW3/hw3/README.md)

以 homography 為核心，自己實作 homography 估計、warping 與 RANSAC（不使用 `cv2.findHomography`、
`cv2.warpPerspective`、`cv2.remap`）。

- Part 1：用 DLT 估計 homography，以 forward warping 合成畫面。
- Part 2：偵測 ArUco marker，以 backward warping 製作 planar AR 影片。
- Part 3：把兩張透視影像反投影回正面，還原其中的 QR code。
- Part 4：ORB 特徵比對、RANSAC 與 homography 串接，把三張影像拼成 panorama。

```bash
cd HW3/hw3/src
bash hw3.sh
```

## Final Project：Stereo Matching

題目說明：[`final_project_stereo_matching_2026.pdf`](<Final Project/final_project_stereo_matching_2026.pdf>)、
[`CV_Final_Project_Rules.pdf`](<Final Project/CV_Final_Project_Rules.pdf>)

從校正過的左右影像計算 disparity map，以 bad pixel ratio 評估：

1. Cost computation：Census transform 與 Hamming distance
2. Cost aggregation：guided filtering
3. Disparity optimization：winner-take-all
4. Disparity refinement：left-right consistency check、hole filling、weighted median filtering

主要程式：`Stereo Matching/computeDisp.py`

```bash
cd "Final Project/Stereo Matching Team_Team18/Stereo Matching"
python main.py --image Tsukuba
```

### Bonus：深度排序與障礙物提示

用手機水平移動拍攝左右影像，經過特徵比對、未校正的 stereo rectification 與 disparity 估計後，排出物體的
遠近順序，並標示為近距離警告、中距離注意或遠距離安全。詳細說明見
[`README_depth_order_bonus.md`](<Final Project/Stereo Matching Team_Team18/Stereo Matching_bonus/README_depth_order_bonus.md>)。

### 後續升級：StereoGuard

期末專題結束後，把 bonus 的手機雙拍深度感知改寫成完整的應用
[StereoGuard](<Final Project/StereoGuard/README.md>)：用預測的焦距做 calibrated rectification，以
FoundationStereo 計算視差，再用 Depth Pro、UniDepthV2、Depth Anything V2 估計手機移動距離換算成公尺；
YOLO11-seg 或 Grounding DINO + SAM 2.1 自動辨識物體，依真實距離門檻給出 DANGER / CAUTION / SAFE 與建議的
行進方向，另外支援影片追蹤、碰撞時間（TTC）估計與 Gradio 網頁介面。在 MiddEval3 上 bad-1.0 從 19.4% 降到
2.6%；已知手機移動距離時，1–3 公尺內的物體距離誤差不到 1 公分。
