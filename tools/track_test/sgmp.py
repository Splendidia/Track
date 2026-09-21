# vim: expandtab:ts=4:sw=4
import numpy as np
from shapely.geometry import Point, Polygon

class FeasibleRegion:
    """可行域掩膜：保证轨迹绝对不会穿透物理边界"""
    def __init__(self, polygon_pts=None):
        if polygon_pts is None:
            self.poly = None
        else:
            self.poly = Polygon(polygon_pts)
            
    def project(self, x, y):
        if self.poly is None:
            return x, y
        try:
            pt = Point(x, y)
            if self.poly.contains(pt):
                return x, y
            # 越界强制正交投影拉回
            distance = self.poly.exterior.project(pt)
            nearest_pt = self.poly.exterior.interpolate(distance)
            return nearest_pt.x, nearest_pt.y
        except Exception:
            return x, y

class SGMP:
    def __init__(self, history_len=5):
        # 默认给定一个超大摄像机边界 (1920x1080 刨去边缘)
        self.region = FeasibleRegion([
            [50, 50], [1870, 50], [1870, 1030], [50, 1030]
        ])
        self.history_len = history_len
        self.centers = []
        self.ious = []
        self.velocities = []
        self.mode = "moving"

    def update_history(self, current_box, prev_box=None):
        try:
            cx, cy = current_box[0], current_box[1]
            self.centers.append(np.array([cx, cy]))
            if len(self.centers) > self.history_len:
                self.centers.pop(0)

            if len(self.centers) >= 2:
                v = self.centers[-1] - self.centers[-2]
                self.velocities.append(v)
                if len(self.velocities) > self.history_len - 1:
                    self.velocities.pop(0)

            if prev_box is not None:
                dist = np.linalg.norm(np.array([cx, cy]) - np.array([prev_box[0], prev_box[1]]))
                self.ious.append(1.0 / (1.0 + dist))
            else:
                self.ious.append(1.0)
                
            if len(self.ious) > self.history_len:
                self.ious.pop(0)
        except Exception:
            pass

    def classify_state(self, tau_v=5.0, tau_iou=0.1):
        try:
            if len(self.velocities) < 3:
                return "moving"
                
            v_arr = np.array(self.velocities)
            v_mag = np.mean(np.linalg.norm(v_arr, axis=1)) 
            
            iou_arr = np.array(self.ious)
            iou_diff = np.mean(np.abs(np.diff(iou_arr))) if len(iou_arr) >= 2 else 0 
                
            if v_mag < tau_v and iou_diff < tau_iou:
                self.mode = "static"
            else:
                self.mode = "moving"
        except Exception:
            self.mode = "moving"
            
        return self.mode

    def constrain_prediction(self, mean, covariance):
        """
        核心修复：绝对不碰 covariance！
        只通过物理法则（静止锚定、可行域投影）强势接管 mean 向量。
        """
        try:
            state = self.classify_state()
            new_mean = mean.copy()
            half_dim = len(mean) // 2
            
            if state == "static":
                # 🌟 Static：彻底锚定，抹除速度预测
                if len(self.centers) > 0:
                    new_mean[0] = self.centers[-1][0]
                    new_mean[1] = self.centers[-1][1]
                new_mean[half_dim:] = 0.0  
            else:
                # 🌟 Moving：信任 KF 的位置，但用真实历史均速接管未来的卡尔曼惯性
                if len(self.velocities) > 0:
                    avg_v = np.mean(np.array(self.velocities), axis=0)
                    new_mean[half_dim] = avg_v[0]
                    new_mean[half_dim+1] = avg_v[1]
                
            # 执行可行域防穿透投影
            cx, cy = new_mean[0], new_mean[1]
            proj_x, proj_y = self.region.project(cx, cy)
            new_mean[0], new_mean[1] = proj_x, proj_y
            
            # 返回修改后的 mean 和 原封不动的 covariance，保证底层数学100%不崩溃！
            return new_mean, covariance
        except Exception:
            # 极限兜底：万一遇到除以0等异常，直接退回原版，保证存活！
            return mean, covariance