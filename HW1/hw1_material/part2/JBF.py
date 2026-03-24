import numpy as np
import cv2

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        """
        Parameters
        ----------
        img : ndarray
            The image to be filtered.

        guidance : ndarray
            The guidance image used to compute range weights.
            It can be either:
              - grayscale: shape (H, W)
              - color:     shape (H, W, C)

        Returns
        -------
        output : ndarray
            The filtered result with the same shape as img.
        """
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int64)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int64)

        ### TODO ###
        # Note: Pixel values should be normalized to [0, 1] (divided by 255) to construct range kernel.
        
        # normalize to [0, 1]
        padded_img = padded_img.astype(np.float64) / 255.0
        padded_guidance = padded_guidance.astype(np.float64) / 255.0

        h, w = img.shape[:2]
        # center guidance: 對應原圖大小的中心區域
        if guidance.ndim == 2:
            center_guidance = padded_guidance[self.pad_w:self.pad_w+h, self.pad_w:self.pad_w+w]
        else:
            center_guidance = padded_guidance[self.pad_w:self.pad_w+h, self.pad_w:self.pad_w+w, :]
        
        # 建 spatial kernel
        ax = np.arange(-self.pad_w, self.pad_w + 1)
        xx, yy = np.meshgrid(ax, ax)
        spatial_kernel = np.exp(-(xx**2 + yy**2) / (2 * (self.sigma_s ** 2)))

        # 累加
        if img.ndim == 2:
            numerator = np.zeros((h, w), dtype=np.float64)
        else:
            numerator = np.zeros((h, w, img.shape[2]), dtype=np.float64)
        
        denominator = np.zeros((h, w), dtype=np.float64)
        
        # 對 window 內每個位移做加權
        for dy in range(-self.pad_w, self.pad_w + 1):
            for dx in range(-self.pad_w, self.pad_w + 1):
                # shifted guidance
                if guidance.ndim == 2:
                    shifted_guidance = padded_guidance[self.pad_w + dy:self.pad_w + dy + h, self.pad_w + dx:self.pad_w + dx + w]
                else:
                    shifted_guidance = padded_guidance[self.pad_w + dy:self.pad_w + dy + h, self.pad_w + dx:self.pad_w + dx + w, :]
                
                # range kernel
                if guidance.ndim == 2:
                    range_kernel = np.exp(-((center_guidance - shifted_guidance) ** 2) / (2 * (self.sigma_r ** 2)))
                else:
                    range_kernel = np.exp(-np.sum((center_guidance - shifted_guidance) ** 2, axis=2) / (2 * (self.sigma_r ** 2)))
                
                # combined kernel
                combined_kernel = spatial_kernel[dy + self.pad_w, dx + self.pad_w] * range_kernel
                
                # shifted image
                shifted_img = padded_img[self.pad_w + dy:self.pad_w + dy + h, self.pad_w + dx:self.pad_w + dx + w]
                
                # accumulate
                numerator += combined_kernel[..., np.newaxis] * shifted_img if img.ndim == 3 else combined_kernel * shifted_img
                denominator += combined_kernel
        # normalize
        if img.ndim == 2:
            output = numerator / denominator
        else:
            output = numerator / denominator[:, :, np.newaxis]
        
        output = output * 255.0    
        return np.clip(output, 0, 255).astype(np.uint8)