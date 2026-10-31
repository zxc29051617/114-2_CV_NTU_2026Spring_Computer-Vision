# HW3: Projective Geometry

這份作業以 **homography（平面投影轉換）** 為核心，依序實作影像 warping、
marker-based planar AR、透視影像校正，以及 panorama stitching。

題目原文請參考 [`../hw3.pdf`](../hw3.pdf)。

## 作業內容

### Part 1: Homography Estimation

目標是由至少四組對應點估計 `3 x 3` homography matrix，並將五張影像以
forward warping 投影到指定 canvas。

主要流程：

1. 使用 Direct Linear Transform（DLT）建立線性方程。
2. 以 SVD 求解 homography。
3. 將 source image 的座標映射至 destination canvas。
4. 依序合成五張影像，輸出 `output1.png`。

入口：[`src/part1.py`](src/part1.py)

### Part 2: Marker-Based Planar AR

目標是偵測影片中的 ArUco marker，將參考影像投影到 marker 平面上，產生
planar AR 影片。

主要流程：

1. 使用 OpenCV ArUco detector 找出 marker corners。
2. 由參考影像四角與 marker 四角估計 homography。
3. 使用 backward warping，避免 forward warping 產生空洞。
4. 逐 frame 寫入影片，輸出 `output2.avi`。

入口：[`src/part2.py`](src/part2.py)

### Part 3: Unwarp the Secret

目標是校正兩張帶有透視變形的影像，還原其中的 QR code。

主要流程：

1. 指定來源影像中的四個 QR code corners。
2. 將 corners 對應到 `500 x 500` 的正方形輸出平面。
3. 估計 homography 並執行 backward warping。
4. 輸出 `output3_1.png` 與 `output3_2.png`。

入口：[`src/part3.py`](src/part3.py)

報告需要比較兩張來源影像的校正結果，並解釋實際影像中的非平面、鏡頭或拍攝
因素為何可能讓 homography 結果不同。

### Part 4: Panorama

目標是將三張連續影像拼接成 panorama。

主要流程：

1. 使用 ORB 偵測 keypoints 與 descriptors。
2. 使用 brute-force matcher 建立特徵對應。
3. 自行實作 RANSAC，排除錯誤 matches 並估計 homography。
4. 將連續影像的 homography 串接到第一張影像的座標系。
5. 使用 backward warping 將影像拼接到共同 canvas。
6. 輸出 `output4.png`。

入口：[`src/part4.py`](src/part4.py)

## 核心程式

[`src/utils.py`](src/utils.py) 包含整份作業共用的兩個核心函式：

### `solve_homography(u, v)`

- 輸入 `N x 2` 的 source points `u` 與 destination points `v`。
- 至少需要四組對應點。
- 建立 `2N x 9` homogeneous linear system。
- 使用 `numpy.linalg.svd()` 取得 null-space solution。
- 回傳正規化後的 `3 x 3` homography matrix。

### `warping(src, dst, H, ..., direction)`

- `direction='f'`：將 source pixels 映射至 destination，適合 Part 1。
- `direction='b'`：逐 destination pixel 反查 source location，適合 Part 2 到 Part 4。
- 目前程式使用 NumPy vectorization 與 nearest-neighbor sampling。
- 實作避免逐 pixel Python for-loop，以符合執行時間限制。

## 資料夾結構

```text
hw3/
├── README.md
├── requirements.txt
├── report_template.docx
├── resource/
│   ├── times.jpg
│   ├── img1.jpg ... img5.jpg
│   ├── hehe.jpg
│   ├── seq0.mp4
│   ├── BL_secret1.png
│   ├── BL_secret2.png
│   └── frame1.jpg ... frame3.jpg
└── src/
    ├── utils.py
    ├── part1.py
    ├── part2.py
    ├── part3.py
    ├── part4.py
    └── hw3.sh
```

`src/` 內可能已有執行後生成的 PNG/AVI；它們是輸出結果，不是演算法入口。

## 環境設定

題目要求 Python 3.6 以上，主要套件如下：

- NumPy
- OpenCV
- OpenCV Contrib（ArUco）
- tqdm

在 `HW3/hw3` 目錄執行：

```bash
pip install -r requirements.txt
```

## 執行方式

程式使用相對路徑讀取 `../resource`，因此應從 `src` 目錄執行：

```bash
cd HW3/hw3/src
bash hw3.sh
```

也可以分別執行：

```bash
python part1.py
python part2.py
python part3.py
python part4.py
```

預期在 `src/` 產生：

```text
output1.png
output2.avi
output3_1.png
output3_2.png
output4.png
```

## 作業限制

依題目規定，不可使用以下函式直接取代指定實作：

- `cv2.findHomography`
- `cv2.warpPerspective`
- `cv2.remap`

其他重要要求：

- Homography 與 warping 必須自行實作。
- Panorama 的 RANSAC 必須自行實作。
- RANSAC 要固定 random seed，確保結果可重現。
- 不可 hard-code 絕對路徑，TA 會在 Linux 執行。
- 全部程式必須在 10 分鐘內完成。
- 提交時不要包含生成的 `*.png` 與 `*.avi`。
