# vim: expandtab:ts=4:sw=4
import cv2
import numpy as np

class DeformableRegistration:
    """
    基于 ORB 特征的可变形配准模块，用于计算两个图像块的结构相似度。
    无需训练，仅使用 OpenCV 传统特征。
    """
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def compute_similarity(self, img_patch1, img_patch2):
        """
        输入两个灰度图像块（numpy数组，uint8），返回结构相似度得分 [0,1]。
        得分越高表示越相似。
        """
        if img_patch1 is None or img_patch2 is None:
            return 0.0
        if img_patch1.size == 0 or img_patch2.size == 0:
            return 0.0

        # 检测关键点和描述子
        kp1, des1 = self.orb.detectAndCompute(img_patch1, None)
        kp2, des2 = self.orb.detectAndCompute(img_patch2, None)

        if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
            return 0.0

        # 特征匹配
        matches = self.bf.knnMatch(des1, des2, k=2)

        # Lowe's 比率测试，筛选好的匹配
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            return 0.0

        # 计算匹配点之间的平均距离（归一化后作为相似度）
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # 计算匹配点对的欧氏距离
        dists = np.linalg.norm(pts1 - pts2, axis=1)
        avg_dist = np.mean(dists)

        # 根据图像块对角线长度归一化
        diag = np.sqrt(img_patch1.shape[0]**2 + img_patch1.shape[1]**2)
        if diag == 0:
            return 0.0
        # 距离越小相似度越高，映射到 [0,1]：相似度 = exp(-avg_dist / diag)
        similarity = np.exp(-avg_dist / diag)
        return similarity

    def extract_patch(self, img, rbox, padding=0.2):
        """
        从图像中提取旋转框对应的图像块，并校正为水平矩形。
        rbox: [cx, cy, w, h, angle] (角度弧度)
        padding: 边距比例，用于包含更多上下文
        返回：校正后的灰度图像块
        """
        cx, cy, w, h, angle = rbox[:5]
        # 增加边距
        w_pad = w * (1 + padding)
        h_pad = h * (1 + padding)

        # 旋转矩阵
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # 计算四个角点
        corners = np.array([[-w_pad/2, -h_pad/2],
                            [ w_pad/2, -h_pad/2],
                            [ w_pad/2,  h_pad/2],
                            [-w_pad/2,  h_pad/2]])
        rotated_corners = corners @ R.T + np.array([cx, cy])

        # 获取包含旋转框的轴对齐边界矩形
        min_xy = np.min(rotated_corners, axis=0).astype(int)
        max_xy = np.max(rotated_corners, axis=0).astype(int)
        min_xy = np.maximum(min_xy, 0)
        max_xy = np.minimum(max_xy, [img.shape[1], img.shape[0]])

        if min_xy[0] >= max_xy[0] or min_xy[1] >= max_xy[1]:
            return None

        # 裁剪图像区域
        roi = img[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0]]

        # 对ROI进行旋转校正，使其水平
        # 计算旋转中心在ROI中的坐标
        center_roi = np.array([cx - min_xy[0], cy - min_xy[1]])
        # 构建旋转矩阵（注意角度取反，因为我们要把旋转框转正）
        rot_mat = cv2.getRotationMatrix2D(tuple(center_roi.astype(float)), -np.degrees(angle), 1.0)
        # 应用仿射变换
        corrected = cv2.warpAffine(roi, rot_mat, (max_xy[0]-min_xy[0], max_xy[1]-min_xy[1]))

        # 转换为灰度
        if len(corrected.shape) == 3:
            corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

        # 截取中间目标区域（去除边距）
        h_roi, w_roi = corrected.shape
        center_h, center_w = h_roi//2, w_roi//2
        crop_h = int(h * (1 + padding/2))
        crop_w = int(w * (1 + padding/2))
        y1 = max(0, center_h - crop_h//2)
        y2 = min(h_roi, center_h + crop_h//2)
        x1 = max(0, center_w - crop_w//2)
        x2 = min(w_roi, center_w + crop_w//2)
        patch = corrected[y1:y2, x1:x2]
        return patch