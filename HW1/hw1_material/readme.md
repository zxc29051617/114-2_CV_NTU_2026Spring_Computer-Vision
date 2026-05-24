DoG.py 說明：

Step 1: Build Gaussian pyramid
首先建立兩個 octaves 的 Gaussian pyramid。每個 octave 共有 5 張 Gaussian images。
在每個 octave 中，第一張影像作為 base image，其餘 4 張影像則分別使用不同的 sigma 進行 Gaussian blur。第二個 octave 的 base image 是由第一個 octave 最後一張 Gaussian image 下採樣 2 倍得到。

Step 2: Build DoG pyramid
接著在每個 octave 中，對相鄰的 Gaussian images 做相減，產生 4 張 DoG images。
也就是令
DoG_i = Gaussian_{i+1} - Gaussian_i。
因此兩個 octaves 總共會得到 8 張 DoG images。

Step 3: Detect keypoints in DoG volume
在每個 octave 中，只檢查中間兩張 DoG images（DoG1 與 DoG2），因為 local extrema 的判斷需要同時比較前後層。
對每一個有效 pixel，忽略影像邊界後，先檢查其絕對值是否大於 threshold；若通過 threshold，再將該點與其 3x3x3 neighborhood 中的 26 個鄰居比較。若該點是 local maximum 或 local minimum，則將其保留為 keypoint。

Step 4: Convert coordinates back to original image scale
若 keypoint 來自第一個 octave，則直接使用其座標 [y, x]；
若 keypoint 來自第二個 octave，由於第二個 octave 的影像尺寸為原圖的一半，因此需將座標乘以 2，轉回原始影像尺度。



Step 5: Remove duplicates and sort keypoints
最後將所有 keypoints 合併，移除重複座標，並按照 row 與 column 進行排序，最後回傳 shape 為 (N, 2) 的 keypoints array。



JBF.py
RGB到Y的mapping：把rgb的每一點mapping到Y上 三維投射到一維 所以不同顏色被MAPPING到相同顏色機率很高
找出一組好的RGB到Y的轉換係數 使得不同顏色轉到灰階後保留原本的顏色區別
bilateral filter特點：除了用spatial kernel 去看空間距離上的差異以外也會使用range kernel 去看像素(顏色之間距離)
對於每個 pixel，看它周圍一塊區域，把距離近而且 guidance 相似的點給大權重，最後做加權平均，輸出新的影像。

joint_bilateral_filter
