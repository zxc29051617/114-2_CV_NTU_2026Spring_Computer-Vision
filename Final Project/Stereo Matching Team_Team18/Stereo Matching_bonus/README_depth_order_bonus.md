# Stereo-based Obstacle Awareness and Depth Ordering Bonus

## 1. Project Overview

This bonus project extends the original stereo matching assignment into a practical depth-perception application.

The original assignment focuses on computing a disparity map from a rectified stereo image pair. In this bonus project, we further use the disparity map to estimate the relative depth order of selected real-world objects and classify them into simple obstacle-awareness levels:

* **DANGER / TOO CLOSE**
* **CAUTION / MID RANGE**
* **SAFE / FAR**

The goal is to demonstrate a low-cost obstacle awareness prototype using only camera images and stereo matching. The system does not require a commercial depth camera or LiDAR sensor. Instead, it uses a mobile phone to capture two views of the same static scene and applies stereo matching to infer relative object depth.

---

## 2. Application Scenario

This system is designed as a simple prototype for low-cost obstacle awareness.

In many practical scenarios, the system may not need to estimate exact metric depth in centimeters. Instead, it may only need to know which object is closer and which object is farther. For example:

* A nearby opaque object may be treated as a potential obstacle.
* A book at a medium distance may be treated as a caution region.
* A far background object may be treated as safe.

Therefore, after computing the disparity map, the system ranks selected object regions by a robust disparity score and assigns each object an obstacle-awareness label. The default score is the 75th percentile disparity among reliable pixels inside each selected object box.

Since larger disparity usually indicates a closer object:

* the object with the largest depth score is labeled **DANGER / TOO CLOSE**
* the object with the middle depth score is labeled **CAUTION / MID RANGE**
* the object with the smallest depth score is labeled **SAFE / FAR**

---

## 3. Method

A single mobile phone is translated horizontally to simulate a stereo camera pair:

* The first photo is treated as the **left view**.
* The second photo is treated as the **right view**.

The complete pipeline is:

1. Capture two phone images from slightly different horizontal positions.
2. Detect and match visual features between the two images.
3. Estimate the fundamental matrix using RANSAC.
4. Perform uncalibrated stereo rectification.
5. Feed the rectified left/right images into `computeDisp.py`.
6. Compute the disparity map.
7. Let the user select object bounding boxes.
8. Compute the reliable-pixel 75th percentile disparity score inside each object box.
9. Rank objects from near to far.
10. Visualize obstacle-awareness labels.

This satisfies the bonus requirement of building a pipeline from image rectification to disparity map computation, and further demonstrates a real-world depth-perception application.

---

## 4. File Structure

The project folder should contain:

```text
Stereo Matching_bonus/
├── computeDisp.py
├── object_depth_order_demo.py
├── make_depth_order_video.py
├── README_depth_order_bonus.md
├── phone_data/
│   └── scene1/
│       ├── left.jpg
│       └── right.jpg
└── results/
```

File descriptions:

```text
computeDisp.py
```

The stereo matching algorithm used to compute the disparity map.

```text
object_depth_order_demo.py
```

Main bonus script. It performs feature matching, rectification, disparity computation, object selection, depth ordering, and obstacle-awareness visualization.

```text
make_depth_order_video.py
```

Optionally creates a short visual demo video from the generated result images. This is only a supplemental visualization and does not replace the recorded presentation slides required by TA.

```text
phone_data/scene1/left.jpg
phone_data/scene1/right.jpg
```

Input stereo pair captured by horizontally translating a mobile phone.

```text
results/depth_order_scene1/
```

Output folder containing intermediate and final results.

---

## 5. Dataset Capture Instructions

Create the following folder:

```text
phone_data/scene1/
├── left.jpg
└── right.jpg
```

Recommended scene setup:

* near object: opaque textured box, mug, toy, or book, about 30 cm from the camera
* middle object: book or box, about 60 cm from the camera
* far object: bookshelf, poster, monitor edge, or other textured background, more than 100 cm away

Capture procedure:

1. Place several objects at different distances.
2. Take the first photo and save it as `left.jpg`.
3. Move the phone horizontally to the right by around 3 to 5 cm.
4. Take the second photo and save it as `right.jpg`.

Important capture rules:

* Do not rotate the phone.
* Do not tilt the phone.
* Do not move the phone up or down.
* Do not move the phone forward or backward.
* Keep the scene static.
* Keep at least 80% overlap between the two images.
* Use normal 1x camera mode.
* Do not use ultra-wide mode.
* Use good lighting and textured objects.
* Avoid transparent bottles, reflective plastic, plain white walls, pure black regions, and large textureless surfaces.

A good scene should contain clear depth layers, such as:

```text
near: textured box
middle: book
far: bookshelf or poster
```

---

## 6. Environment Setup

Install the required packages:

```bash
pip install -r requirements_bonus.txt
```

The package `opencv-contrib-python` is required because `computeDisp.py` uses OpenCV extra modules such as `cv2.ximgproc`.

If Python reports `No module named 'cv2.ximgproc'`, remove non-contrib OpenCV wheels and install the contrib build:

```bash
python -m pip uninstall opencv-python opencv-python-headless opencv-contrib-python-headless -y
python -m pip install opencv-contrib-python
```

---

## 7. Run the Bonus Demo

Run the main bonus script:

```bash
python object_depth_order_demo.py --left phone_data/scene1/left.jpg --right phone_data/scene1/right.jpg --out_dir results/depth_order_scene1 --max_disp 96
```

After running the command, a window will appear for ROI selection.

---

## 8. How to Select Object Boxes

When the image window appears, select three objects using the mouse.

Recommended selection:

1. Select the near object, such as an opaque box, mug, toy, or book cover.
2. Select the middle object, such as a book.
3. Select the far object, such as a wall, shelf, or monitor.

Operation steps:

```text
Draw a box around the first object  → press ENTER or SPACE
Draw a box around the second object → press ENTER or SPACE
Draw a box around the third object  → press ENTER or SPACE
Press ESC to finish all selections
```

Important:

* Press `ENTER` or `SPACE` after each selected object.
* Press `ESC` only after all objects have been selected.
* Do not press `c` unless you want to cancel the current selection.

The selection order does not determine the final depth order. The program ranks the objects automatically according to their reliable-pixel p75 disparity score.

---

## 9. Outputs

After execution, the following files are generated:

```text
results/depth_order_scene1/
├── matches.png
├── rectified_left.png
├── rectified_right.png
├── rectified_pair_guides.png
├── disparity_raw.png
├── disparity_color.png
├── object_depth_order_result.png
├── obstacle_warning_result.png
└── depth_order.csv
```

Output descriptions:

```text
matches.png
```

Shows feature correspondences between the two phone images.

```text
rectified_left.png
rectified_right.png
```

The rectified left and right images after uncalibrated stereo rectification.

```text
rectified_pair_guides.png
```

Shows the rectified image pair with horizontal guide lines. This is used to check whether corresponding points are horizontally aligned.

```text
disparity_raw.png
```

Raw disparity map computed by `computeDisp.py`.

```text
disparity_color.png
```

Colored visualization of the disparity map. Larger disparity generally indicates closer objects.

```text
disparity_reliable_raw.png
disparity_reliable_color.png
```

Reliable disparity pixels used for object ranking. These pixels pass the left-right consistency check and a winner-cost margin check before ROI scoring. This reduces false scores from textureless walls and hole-filled regions.

```text
object_depth_order_result.png
```

Shows the selected objects ranked by reliable-pixel p75 disparity score.

```text
obstacle_warning_result.png
```

The main application result. It shows object boxes with obstacle-awareness labels:

* red box: **DANGER / TOO CLOSE**
* yellow box: **CAUTION / MID RANGE**
* green box: **SAFE / FAR**

```text
depth_order.csv
```

Stores the numerical result, including object rank, depth score, p50 disparity, p75 disparity, reliable-pixel ratio, whether the ROI score was used, and bounding box coordinates.

---

## 10. Optional Demo Video

After generating the result images, an optional demo video can be created:

```bash
python make_depth_order_video.py --result_dir results/depth_order_scene1 --output depth_order_demo_video.mp4
```

The video demonstrates the complete pipeline as an optional visual aid:

1. Feature matching
2. Stereo rectification
3. Disparity map computation
4. Object depth ordering
5. Obstacle-awareness result

The optional output video is:

```text
depth_order_demo_video.mp4
```

---

## 11. Example Console Output

An example output may look like:

```text
Depth order, near to far (score = reliable p75 disparity):
1. Object_1: depth score = 39.00, p50 = 31.00, p75 = 39.00, valid = 35.2%
2. Object_3: depth score = 19.00, p50 = 16.00, p75 = 19.00, valid = 28.4%
3. Object_2: depth score = 15.00, p50 = 12.00, p75 = 15.00, valid = 19.1%
Saved results to: results/depth_order_scene1
```

This means:

```text
Object_1 has the largest depth score, so it is closest.
Object_3 is at the middle distance.
Object_2 has the smallest depth score, so it is farthest.
```

The final visualization will map these results to:

```text
closest object  → DANGER / TOO CLOSE
middle object   → CAUTION / MID RANGE
farthest object → SAFE / FAR
```

---

## 12. Notes and Limitations

This bonus system is a prototype. It estimates relative object depth, not precise metric distance.

Possible failure cases include:

* too much camera rotation between the two phone images
* too little overlap between left and right images
* textureless objects or plain walls
* transparent or reflective objects, such as clear water bottles
* moving objects during capture
* strong lighting changes
* inaccurate ROI selection
* poor rectification quality

To improve results:

* move the phone only horizontally
* keep the phone orientation fixed
* use a small baseline of around 3 to 5 cm
* include textured objects in the scene
* choose opaque objects instead of transparent or reflective objects
* select only textured and reliable regions inside each object, not the whole object plus background
* avoid ultra-wide camera mode
* ensure the selected boxes contain valid disparity regions

---

## 13. Report Wording

We extend the original stereo matching task to a practical depth-perception application: stereo-based obstacle awareness and depth ordering. A mobile phone is horizontally translated to simulate a stereo pair. We estimate epipolar geometry from feature correspondences, rectify the two views, and compute a disparity map using our existing `computeDisp.py` algorithm.

For each selected object region, we compute a robust reliable-pixel p75 disparity score and rank objects from near to far. Since larger disparity indicates a closer object, the system assigns obstacle-awareness labels to the selected objects: **DANGER / TOO CLOSE**, **CAUTION / MID RANGE**, and **SAFE / FAR**.

This bonus demonstrates how stereo matching can be used not only to estimate disparity maps, but also to support a simple low-cost obstacle awareness application.
