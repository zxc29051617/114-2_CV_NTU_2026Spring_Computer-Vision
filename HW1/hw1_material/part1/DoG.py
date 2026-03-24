import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        """
        Detect DoG keypoints from a grayscale image.

        Args:
            image (np.ndarray): Input grayscale image with shape (H, W) (np.float64).

        Returns:
            keypoints (np.ndarray): Array with shape (N, 2)
        """
        # TODO:
        # Step 1: Build Gaussian pyramid (2 octaves, 5 Gaussian images per octave)
        # - Use cv2.GaussianBlur(src, ksize=(0, 0), sigmaX=self.sigma**k). 
        # - Octave 1 image shape: (H, W).
        # - Octave 2 base image: downsample octave 1's last image (sigma**4) by 2 using
        #   cv2.resize(..., interpolation=cv2.INTER_NEAREST),
        #   so shape becomes (H//2, W//2).
        gaussian_images = [] # 準備存兩個 octave 的 Gaussian 圖
        current_base = image.astype(np.float64).copy() # 一開始，第一個 octave 的 base 就是原圖
        for octave in range(self.num_octaves): #因為有 2 個 octave，所以會跑兩次
            octave_gaussians = [] # 用來存 這一個 octave 的 5 張 Gaussian image
            octave_gaussians.append(current_base)

            for k in range(1, self.num_guassian_images_per_octave): # 每個 octave 要產生 5 張 Gaussian image，所以 k 從 0 跑到 4
                blurred = cv2.GaussianBlur(
                    current_base,
                    ksize=(0, 0),
                    sigmaX=self.sigma ** k # sigmaX 的值會隨著 k 的增加而增加，這樣就能得到不同模糊程度的圖像
                )
                octave_gaussians.append(blurred)

            gaussian_images.append(octave_gaussians) # 把這個 octave 的 5 張圖存起來

            if octave != self.num_octaves - 1:
                last_img = octave_gaussians[-1] # 取這個 octave 最後一張，最模糊那張
                h, w = last_img.shape
                current_base = cv2.resize(  # 把它縮小成一半，作為下一個 octave 的 base
                    last_img,
                    (w // 2, h // 2),
                    interpolation=cv2.INTER_NEAREST
                )
#    print("num octaves =", len(gaussian_images))
#    for idx in range(len(gaussian_images)):
#        print(f"octave {idx} has {len(gaussian_images[idx])} gaussian images")

        # 做完 Step 1 之後，結構：
        # gaussian_images = [
        #   [G0, G1, G2, G3, G4],   # octave 0
        #   [G0, G1, G2, G3, G4]    # octave 1
        #   ]

        # Step 2: Build DoG pyramid (2 octaves, 4 DoG images per octave)
        # - For each octave, subtract adjacent Gaussian images:
        #   DoG_i = Gaussian_{i+1} - Gaussian_i.
        # - Use cv2.subtract(second_image, first_image).
        # - DoG image shape is the same as its corresponding Gaussian image.

        # 做完 Step 2 之後，結構：
        # dog_images = [
        #   [D0, D1, D2, D3],   # octave 0 
        #   [D0, D1, D2, D3]    # octave 1
        #   D0 = G1 - G0, D1 = G2 - G1, D2 = G3 - G2, D3 = G4 - G3
        #   ]
        dog_images = []
        for octave in range(self.num_octaves):
            octave_dogs = []
            for i in range(self.num_DoG_images_per_octave):
                dog = cv2.subtract(
                    gaussian_images[octave][i + 1],
                    gaussian_images[octave][i]
                )
                octave_dogs.append(dog)
            dog_images.append(octave_dogs)
        # print(len(dog_images))          # 應該是 2
        # print(len(dog_images[0]))       # 應該是 4
        # print(len(dog_images[1]))       # 應該是 4
        # print(dog_images[0][0].shape)   # 應該是 (H, W)
        # print(dog_images[1][0].shape)   # 應該是 (H//2, W//2)

        # Step 3: Threshold and find 3D local extrema in DoG volume
        # - Ignore 1-pixel image border.
        # - For each valid pixel in DoG images 1,2 of each octave, compare against
        #   its 26 neighbors in a 3x3x3 neighborhood. 
        #   假設你現在在檢查某個點 (y, x)，而它位於中間那層 DoG。 你要看一個 3 x 3 x 3 的小立方體：但中間那個位置就是自己，所以不能拿自己跟自己比。所以最後要比的是：26 個鄰居
        # - Keep [y, x] as a keypoint if:
        #   (1) it is a local maximum or minimum (>= max or <= min), and
        #   (2) abs(DoG value) > self.threshold.
        # - Coordinates stored in keypoints must be in original image scale:
        #   octave 1 -> [y, x], octave 2 -> [2*y, 2*x].

        # 找 keypoints
        keypoints = []
        for octave in range(self.num_octaves):
            for dog_idx in range(1, self.num_DoG_images_per_octave - 1): # 因為要找 3x3x3 的 neighborhood，所以 dog_idx 只能從 1 到 2，不能是 0 或 3
                prev_img = dog_images[octave][dog_idx - 1]
                curr_img = dog_images[octave][dog_idx]
                next_img = dog_images[octave][dog_idx + 1]
                
                h, w = curr_img.shape

                for y in range(1, h - 1):
                    for x in range(1, w - 1):
                        val = curr_img[y, x]

                        # threshold 先過濾弱 response
                        if abs(val) <= self.threshold:
                            continue

                    # 取 3x3x3 neighborhood
                        neighborhood = np.concatenate([
                            prev_img[y - 1:y + 2, x - 1:x + 2].flatten(),
                            curr_img[y - 1:y + 2, x - 1:x + 2].flatten(),
                            next_img[y - 1:y + 2, x - 1:x + 2].flatten()
                        ])

                        # 把自己刪掉，留下 26 個鄰居
                        center_index = 9 + 4
                        neighborhood = np.delete(neighborhood, center_index)

                        # 檢查 local maximum / minimum
                        if val > np.max(neighborhood) or val < np.min(neighborhood):
                            if octave == 0:
                                keypoints.append([y, x])
                            else:
                                keypoints.append([2 * y, 2 * x])

        # Step 4: Remove duplicate keypoints
        # - Use np.unique(..., axis=0).
        # - Expected shape after this step: (N, 2).
        if len(keypoints) == 0:  #「完全沒找到點」的情況
            return np.zeros((0, 2), dtype=np.int32) # 回傳一個 shape 是 (0, 2) 的空陣列，表示沒有 keypoints

        keypoints = np.array(keypoints, dtype=np.int32)
        keypoints = np.unique(keypoints, axis=0)

        

        # Sort points using np.lexsort((col, row)) -> primary key col, secondary key row.
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        return keypoints