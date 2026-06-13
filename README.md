# NTU Computer Vision 2026 Spring

這個 repository 收錄臺大 114-2 Computer Vision 課程的講義、Python/OpenCV
教學、三份作業，以及 Stereo Matching 期末專題。

內容從基礎影像處理與局部特徵，延伸到場景分類、CNN、projective geometry、
panorama、stereo matching 與深度感知。

## 專案結構

```text
CV/
├── README.md
├── lec00.pdf ... lec16.pdf       # 課程講義
├── SfM.pdf                       # Structure from Motion 補充教材
├── CV_Course_Tutorial/           # Python、NumPy、OpenCV 入門
├── HW1/                          # DoG 與 Joint Bilateral Filter
├── HW2/                          # 場景辨識與 CNN 影像分類
├── HW3/                          # Homography、AR、Panorama
└── Final Project/                # Stereo Matching 與深度感知應用
```

大型影像資料、模型權重、實驗結果與輸出檔也保存在各作業目錄內，因此實際目錄
會比上面的導覽更完整。

## 建議閱讀順序

1. 先閱讀本文件，確認每份作業的目標與入口。
2. Python、NumPy 或 OpenCV 不熟時，先看
   [`CV_Course_Tutorial/README.md`](CV_Course_Tutorial/README.md)。
3. 開始作業前，先讀該作業的題目 PDF，再看程式入口，最後才深入演算法檔案。
4. 建議依序閱讀 `HW1 -> HW2 -> HW3 -> Final Project`，概念會逐步銜接。

## 教學與講義

### CV Course Tutorial

這是課程提供的 Python、NumPy 與 OpenCV 入門教材。

- Python 基礎：[`python_tutorial.ipynb`](CV_Course_Tutorial/python_tutorial.ipynb)
- NumPy 基礎：[`numpy_tutorial.ipynb`](CV_Course_Tutorial/numpy_tutorial.ipynb)
- OpenCV 範例：[`OpenCV/`](CV_Course_Tutorial/OpenCV)
- 安裝說明：[`Python Intro & Install.pdf`](<CV_Course_Tutorial/Python Intro & Install.pdf>)

第一次接觸此專案時，建議從
[`CV_Course_Tutorial/README.md`](CV_Course_Tutorial/README.md) 開始。

### Lecture Notes

根目錄的 `lec00.pdf` 到 `lec16.pdf` 是課堂講義；`SfM.pdf` 是 Structure from
Motion 相關教材。若要先理解程式背後的理論，應先查閱對應講義。

## HW1: DoG 與 Joint Bilateral Filter

題目說明：[`HW1/hw1_2026_spring.pdf`](HW1/hw1_2026_spring.pdf)  
作業筆記：[`HW1/hw1_material/readme.md`](HW1/hw1_material/readme.md)

HW1 練習基礎影像處理、尺度空間與邊緣保留濾波。

### Part 1: Difference of Gaussian

- 建立兩個 octave 的 Gaussian pyramid。
- 對相鄰 Gaussian images 相減，建立 DoG pyramid。
- 在 `3 x 3 x 3` 鄰域中尋找局部極值。
- 將不同 octave 的 keypoint 座標轉回原始影像尺度。
- 移除重複 keypoints 並排序。

閱讀順序：

1. [`part1/main.py`](HW1/hw1_material/part1/main.py)：執行入口與結果輸出。
2. [`part1/DoG.py`](HW1/hw1_material/part1/DoG.py)：Gaussian/DoG pyramid 與 keypoint 偵測。
3. [`part1/eval.py`](HW1/hw1_material/part1/eval.py)：結果評估。

### Part 2: Joint Bilateral Filter

- 實作 bilateral filter 與 joint bilateral filter。
- spatial kernel 衡量空間距離，range kernel 衡量 guidance 顏色差異。
- 比較不同 RGB-to-grayscale 權重產生的 guidance image。
- 找出最能保留原始顏色差異的灰階轉換。

閱讀順序：

1. [`part2/main.py`](HW1/hw1_material/part2/main.py)：讀取設定、建立 guidance 並輸出結果。
2. [`part2/JBF.py`](HW1/hw1_material/part2/JBF.py)：Joint Bilateral Filter 核心實作。
3. [`part2/eval.py`](HW1/hw1_material/part2/eval.py)：結果評估。

## HW2: Recognition and Classification

題目說明：[`HW2/cv2026_hw2.pdf`](HW2/cv2026_hw2.pdf)  
環境與資料集：[`HW2/hw2/README.md`](HW2/hw2/README.md)

HW2 包含傳統手工特徵場景辨識，以及深度學習影像分類兩部分。

### Part 1: Bag-of-Words Scene Recognition

- Tiny Image feature。
- Dense SIFT 與 visual vocabulary。
- K-means 建立 visual words。
- Bag-of-SIFT histogram。
- 自行實作 nearest-neighbor classifier。
- 產生 accuracy 與 confusion matrix。

閱讀順序：

1. [`p1/p1.py`](HW2/hw2/p1/p1.py)：資料、特徵、分類與評估的完整流程。
2. [`p1/utils.py`](HW2/hw2/p1/utils.py)：Tiny Image、SIFT vocabulary、BoW 與分類器。
3. [`p1/p1_run.sh`](HW2/hw2/p1/p1_run.sh)：執行範例。

### Part 2: CNN Image Classification

- 自行建立 CNN `MyNet`。
- 建立並調整 `ResNet18`。
- 完成 train/validation、logging、learning curves 與 best checkpoint。
- 載入 checkpoint 執行 inference，輸出 `filename,label` CSV。
- 使用官方 evaluation script 計算 accuracy。

閱讀順序：

1. [`p2/p2_train.py`](HW2/hw2/p2/p2_train.py)：訓練與驗證主流程。
2. [`p2/model.py`](HW2/hw2/p2/model.py)：`MyNet` 與 `ResNet18` 架構。
3. [`p2/dataset.py`](HW2/hw2/p2/dataset.py)：Dataset、DataLoader 與 augmentation。
4. [`p2/config.py`](HW2/hw2/p2/config.py)：epochs、batch size、optimizer 與 learning rate。
5. [`p2/p2_inference.py`](HW2/hw2/p2/p2_inference.py)：checkpoint 載入與 CSV 輸出。
6. [`p2/p2_eval.py`](HW2/hw2/p2/p2_eval.py)：官方評估流程，不應修改。
7. [`p2/utils.py`](HW2/hw2/p2/utils.py)：log、CSV 與 JSON 工具，不應修改。

## HW3: Projective Geometry

題目說明：[`HW3/hw3.pdf`](HW3/hw3.pdf)  
作業說明：[`HW3/hw3/README.md`](HW3/hw3/README.md)

HW3 以 homography 為核心，練習影像投影、AR 與 panorama。

### 四個部分

- Part 1：使用 DLT 估計 homography，並以 forward warping 合成 canvas。
- Part 2：偵測 ArUco marker，以 backward warping 製作 planar AR 影片。
- Part 3：反投影兩張透視影像，還原其中的 QR code。
- Part 4：使用 ORB feature matching、自行實作 RANSAC 與 homography chaining，
  將三張影像拼接成 panorama。

閱讀順序：

1. [`src/utils.py`](HW3/hw3/src/utils.py)：`solve_homography()` 與 `warping()` 核心。
2. [`src/part1.py`](HW3/hw3/src/part1.py)：Homography 與 forward warping。
3. [`src/part2.py`](HW3/hw3/src/part2.py)：Marker-based planar AR。
4. [`src/part3.py`](HW3/hw3/src/part3.py)：透視校正與 QR code 還原。
5. [`src/part4.py`](HW3/hw3/src/part4.py)：Feature matching、RANSAC 與 panorama。
6. [`src/hw3.sh`](HW3/hw3/src/hw3.sh)：TA 使用的整體執行入口。

作業限制包括不可直接使用 `cv2.findHomography`、`cv2.warpPerspective` 或
`cv2.remap` 取代指定實作。

## Final Project: Stereo Matching

題目說明：

- [`final_project_stereo_matching_2026.pdf`](<Final Project/final_project_stereo_matching_2026.pdf>)
- [`CV_Final_Project_Rules.pdf`](<Final Project/CV_Final_Project_Rules.pdf>)

期末專題從 rectified stereo image pair 計算 disparity map，並以 bad pixel ratio
評估。核心流程包含：

1. Cost computation：目前使用 Census transform 與 Hamming distance。
2. Cost aggregation：使用 guided filtering 平滑 cost volume。
3. Disparity optimization：以 winner-take-all 選擇 disparity。
4. Disparity refinement：left-right consistency check、hole filling 與 weighted
   median filtering。

閱讀順序：

1. [`Stereo Matching/main.py`](<Final Project/Stereo Matching Team_Team18/Stereo Matching/main.py>)：
   讀取 stereo pair、呼叫演算法並執行評估。
2. [`Stereo Matching/computeDisp.py`](<Final Project/Stereo Matching Team_Team18/Stereo Matching/computeDisp.py>)：
   disparity estimation 核心實作。
3. [`Stereo Matching/eval.py`](<Final Project/Stereo Matching Team_Team18/Stereo Matching/eval.py>)：
   bad pixel ratio 評估。

### Bonus: Depth Ordering and Obstacle Awareness

Bonus 使用手機水平位移拍攝左右影像，完成 feature matching、uncalibrated stereo
rectification、disparity estimation 與物體深度排序，再將物體標示為近距離警告、
中距離注意或遠距離安全。

詳細流程與執行方式請看
[`README_depth_order_bonus.md`](<Final Project/Stereo Matching Team_Team18/Stereo Matching_bonus/README_depth_order_bonus.md>)。

## 最快起手方式

若只想快速理解整個 repository：

1. 先看各作業題目 PDF 的 outline/assignment description。
2. HW1 從 `part1/main.py` 開始。
3. HW2 傳統方法從 `p1/p1.py` 開始，CNN 從 `p2/p2_train.py` 開始。
4. HW3 先看 `src/utils.py`，再依序讀 `part1.py` 到 `part4.py`。
5. Final Project 先看 `Stereo Matching/main.py`，再讀 `computeDisp.py`。
