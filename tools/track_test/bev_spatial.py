import numpy as np
import cv2

class BEVSpatial:
    """
    Pseudo-BEV Topology Mapping
    负责宏观防线：将 2D 像素坐标映射到伪 3D 鸟瞰图物理平面。
    """
    @staticmethod
    def rbox_to_polygon(rbox):
        rbox = np.array(rbox, dtype=np.float32).flatten()
        x, y, w, h, angle = rbox[:5]
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        half_w, half_h = w / 2, h / 2
        corners = np.array([
            [-half_w, -half_h], [half_w, -half_h],
            [half_w, half_h], [-half_w, half_h]
        ])
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = np.dot(corners, rot_mat.T)
        rotated[:, 0] += x
        rotated[:, 1] += y
        return rotated.flatten()

    @staticmethod
    def ensure_8point(box):
        box = np.array(box, dtype=np.float32).flatten()
        return BEVSpatial.rbox_to_polygon(box) if len(box) == 5 else box

    @classmethod
    def compute_bev_similarity(cls, box_a, box_b):
        """计算两个框在 3D 物理平面上的拓扑相似度"""
        pts_a = cls.ensure_8point(box_a).reshape(4, 2)
        pts_b = cls.ensure_8point(box_b).reshape(4, 2)
        
        # 提取脚印坐标（底部中心）
        pts_a = pts_a[np.argsort(pts_a[:, 1])]
        pts_b = pts_b[np.argsort(pts_b[:, 1])]
        foot_a = np.mean(pts_a[2:], axis=0)
        foot_b = np.mean(pts_b[2:], axis=0)
        
        # 尺度归一化
        scale = (np.linalg.norm(np.max(pts_a, axis=0) - np.min(pts_a, axis=0)) + 
                 np.linalg.norm(np.max(pts_b, axis=0) - np.min(pts_b, axis=0))) / 2.0 + 1e-5
        
        # 物理距离计算（Y 轴增强映射）
        dx = foot_a[0] - foot_b[0]
        dy = (foot_a[1] - foot_b[1]) * 2.5 
        bev_dist = np.sqrt(dx**2 + dy**2)
        
        return max(0.0, 1.0 - (bev_dist / (1.5 * scale)))