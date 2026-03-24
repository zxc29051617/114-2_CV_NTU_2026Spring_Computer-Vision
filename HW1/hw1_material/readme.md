DoG.py

Step 1：做 Gaussian pyramid

每個 octave 做 5 張模糊圖

Step 2：做 DoG pyramid

每個 octave 的相鄰 Gaussian 相減，得到 4 張 DoG

Step 3：找 keypoints

在每個 octave 的 DoG1、DoG2 裡，對每個 pixel：

忽略邊界
檢查 abs(value) > threshold
跟 26 鄰居比大小
是 local max / min 就留下
Step 4：座標轉回原圖尺度

第二個 octave 的點乘 2

Step 5：去重複 + 排序

最後回傳 (N, 2)



JBF.py
RGB到Y的mapping：把rgb的每一點mapping到Y上 三維投射到一維 所以不同顏色被MAPPING到相同顏色機率很高
找出一組好的RGB到Y的轉換係數 使得不同顏色轉到灰階後保留原本的顏色區別
bilateral filter特點：除了用spatial kernel 去看空間距離上的差異以外也會使用range kernel 去看像素(顏色之間距離)
對於每個 pixel，看它周圍一塊區域，把距離近而且 guidance 相似的點給大權重，最後做加權平均，輸出新的影像。

joint_bilateral_filter
