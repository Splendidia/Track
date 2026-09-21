# vim: expandtab:ts=4:sw=4
import cv2
import numpy as np

class FQAGate:
    """
    Feature Quality-Aware Gate (FQA-Gate)
    负责微观防线的图像预处理、极限降维打击与质量感知评分。
    """
    def __init__(self, fqa_scale=0.125, iqe_threshold=0.40, area_threshold=500):
        self.fqa_scale = fqa_scale
        self.iqe_threshold = iqe_threshold
        self.area_threshold = area_threshold
        self._cached_img_id = None
        self._cached_scaled_img = None

    def fast_downsample(self, img):
        """全局单例缓存缩放：每帧只执行一次极其高效的最近邻插值"""
        if self._cached_img_id != id(img):
            self._cached_scaled_img = cv2.resize(
                img, (0, 0), fx=self.fqa_scale, fy=self.fqa_scale, interpolation=cv2.INTER_NEAREST
            )
            self._cached_img_id = id(img)
        return self._cached_scaled_img

    def scale_box(self, box):
        """将原图坐标缩放至 0.125x 尺度"""
        small_box = np.array(box, dtype=np.float32)
        small_box[:4] = small_box[:4] * self.fqa_scale
        return small_box

    def check_quality(self, img_gray, rbox):
        """
        极速质量评估 (IQE): 
        返回 (is_valid, iqe_score)。若质量达标，才允许提取底层特征。
        """
        cx, cy, w, h = rbox[0], rbox[1], rbox[2], rbox[3]
        
        # 1. 微小目标免检策略 (Area Gating)
        if w * h <= self.area_threshold:
            return False, 0.0

        # 2. 廉价切片抠图
        x1, y1 = max(0, int(cx - w/2)), max(0, int(cy - h/2))
        x2, y2 = min(img_gray.shape[1], int(cx + w/2)), min(img_gray.shape[0], int(cy + h/2))
        fast_patch = img_gray[y1:y2, x1:x2]

        if fast_patch.size == 0:
            return False, 0.0

        gray_patch = fast_patch if len(fast_patch.shape) == 2 else cv2.cvtColor(fast_patch, cv2.COLOR_BGR2GRAY)
        
        # 3. 缩略图极速算分策略 (Thumbnail Laplacian)
        if gray_patch.shape[0] > 32 or gray_patch.shape[1] > 32:
            gray_patch = cv2.resize(gray_patch, (32, 32), interpolation=cv2.INTER_NEAREST)

        mean_brightness = np.mean(gray_patch) / 255.0
        laplacian_var = cv2.Laplacian(gray_patch, cv2.CV_64F).var()
        contrast_score = min(laplacian_var / 500.0, 1.0)
        
        iqe_score = np.clip(0.5 * mean_brightness + 0.5 * contrast_score, 0.1, 0.9)

        # 4. 门控判定
        is_valid = iqe_score > self.iqe_threshold
        return is_valid, iqe_score