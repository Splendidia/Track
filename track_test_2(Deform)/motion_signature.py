import cv2
import numpy as np

class MotionSignatureExtractor:
    """
    运动签名提取器（简化版，不使用PCA）
    每个目标维护一个固定长度的特征向量作为签名
    """
    def __init__(self, max_keypoints=15):
        self.max_keypoints = max_keypoints
        self.signature = None          # 当前签名（原始特征向量）
        self.is_initialized = False    # 标记是否已有签名

    def extract_raw_features(self, prev_gray, curr_gray, prev_bbox, curr_bbox):
        """
        从前后两帧的旋转框中提取原始运动特征，返回固定长度的特征向量。
        如果关键点不足，则用最后一个点重复填充。
        """
        # 检查图像有效性
        if prev_gray is None or curr_gray is None:
            return None
        if prev_gray.shape != curr_gray.shape:
            print(f"Warning: image shape mismatch: prev {prev_gray.shape} vs curr {curr_gray.shape}")
            return None

        # 生成旋转框掩码
        mask = np.zeros_like(prev_gray, dtype=np.uint8)
        points = cv2.boxPoints(((prev_bbox[0], prev_bbox[1]),
                                (prev_bbox[2], prev_bbox[3]),
                                np.degrees(prev_bbox[4])))
        points = np.int32(points)
        cv2.fillPoly(mask, [points], 255)

        # Shi-Tomasi角点检测
        corners = cv2.goodFeaturesToTrack(prev_gray,
                                          maxCorners=self.max_keypoints,
                                          qualityLevel=0.01,
                                          minDistance=5,
                                          mask=mask)
        if corners is None or len(corners) < 5:
            return None
        prev_pts = corners.reshape(-1, 2).astype(np.float32)

        # LK光流跟踪到当前帧
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray,
                                                        prev_pts, None)
        status = status.flatten()  # 将 (N,1) 转为 (N,)
        good_prev = prev_pts[status == 1]
        good_next = next_pts[status == 1]
        if len(good_prev) < 5:
            return None

        # 固定点数：如果超过 max_keypoints，随机采样；如果不足，重复最后一个点填充
        if len(good_prev) > self.max_keypoints:
            idx = np.random.choice(len(good_prev), self.max_keypoints, replace=False)
            good_prev = good_prev[idx]
            good_next = good_next[idx]
        elif len(good_prev) < self.max_keypoints:
            pad = self.max_keypoints - len(good_prev)
            last_prev = good_prev[-1:]
            last_next = good_next[-1:]
            good_prev = np.concatenate([good_prev, np.repeat(last_prev, pad, axis=0)], axis=0)
            good_next = np.concatenate([good_next, np.repeat(last_next, pad, axis=0)], axis=0)

        # 计算关键点相对于目标中心的位置和位移
        prev_center = np.array([prev_bbox[0], prev_bbox[1]])
        curr_center = np.array([curr_bbox[0], curr_bbox[1]])
        rel_pos_prev = good_prev - prev_center
        rel_pos_curr = good_next - curr_center
        displacement = good_next - good_prev

        # 归一化（除以目标尺度）
        scale = max(prev_bbox[2], prev_bbox[3])
        if scale < 1e-5:
            return None
        rel_pos_prev_norm = rel_pos_prev / scale
        rel_pos_curr_norm = rel_pos_curr / scale
        displacement_norm = displacement / scale

        # 拼接成特征向量
        features = np.concatenate([rel_pos_prev_norm.flatten(),
                                    rel_pos_curr_norm.flatten(),
                                    displacement_norm.flatten()])
        return features

    def update_signature(self, raw_feat):
        """用新特征更新签名（直接保存）"""
        if raw_feat is not None:
            self.signature = raw_feat.copy()
            self.is_initialized = True

    def compute_distance(self, raw_feat):
        """计算当前签名与给定特征的距离"""
        if not self.is_initialized or self.signature is None or raw_feat is None:
            return 1.0   # 最大距离
        # 计算欧氏距离
        dist = np.linalg.norm(self.signature - raw_feat)
        sim = np.linalg.norm(self.signature - raw_feat)
        return dist