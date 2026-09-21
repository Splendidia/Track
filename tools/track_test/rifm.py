# vim: expandtab:ts=4:sw=4
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

class RIFMExtractor:
    def __init__(self, lbp_points=6, lbp_radius=1):
        self.lbp_points = lbp_points
        self.lbp_radius = lbp_radius

    def extract_features(self, patch):
        """提取旋转不变的形状和纹理特征"""
        if patch is None or patch.size == 0:
            return None
        
        if len(patch.shape) == 3:
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch

        h, w = gray.shape
        if h > 10 and w > 10:
            ch, cw = int(h * 0.2), int(w * 0.2)
            gray = gray[ch:h-ch, cw:w-cw]
        # 🚀 极限加速秘诀：统一缩小到 32x32，抛弃冗余像素，专门提取宏观纹理和形状！
        gray = cv2.resize(gray, (32, 32))

        # 1. 形状特征：Hu 矩
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-12)

        # 2. 纹理特征：旋转不变 LBP
        lbp = local_binary_pattern(gray, self.lbp_points, self.lbp_radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, self.lbp_points + 3), range=(0, self.lbp_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)

        return {"hu": hu_moments, "lbp": hist}

    def compute_similarity(self, feat1, feat2):
        if feat1 is None or feat2 is None:
            return 0.0

        hu_dist = np.linalg.norm(feat1["hu"] - feat2["hu"])
        hu_sim = np.exp(-hu_dist * 0.5)

        lbp_dist = cv2.compareHist(
            feat1["lbp"].astype(np.float32), 
            feat2["lbp"].astype(np.float32), 
            cv2.HISTCMP_BHATTACHARYYA
        )
        lbp_sim = max(0, 1.0 - lbp_dist)

        # return 0.4 * hu_sim + 0.6 * lbp_sim
        return 0.1 * hu_sim + 0.9 * lbp_sim