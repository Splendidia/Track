# # vim: expandtab:ts=4:sw=4
# """
# 运动记忆增强模块
# 为每个轨迹维护运动模式记忆库，用于计算当前观测与历史运动模式的一致性
# 完全无需训练，仅使用经典计算机视觉方法
# """
# import cv2
# import numpy as np

# class MotionMemory:
#     """
#     运动模式记忆库
#     存储轨迹的历史运动特征，并提供一致性计算方法
#     """
#     def __init__(self, max_length=10):
#         self.max_length = max_length          # 记忆库最大长度
#         self.flow_features = []                # 光流特征列表
#         self.displacements = []                 # 位移向量列表 (dx, dy)
#         self.angle_changes = []                 # 角度变化列表
#         self.timestamps = []                     # 对应帧号
#         self.bboxes = []                          # 对应检测框（可选）

#     def add(self, flow_feat, disp, angle_delta, frame_id, bbox=None):
#         """
#         添加新的运动观测到记忆库
#         flow_feat: 光流特征向量 (numpy数组)
#         disp: 位移向量 (dx, dy)
#         angle_delta: 角度变化 (弧度)
#         frame_id: 当前帧号
#         bbox: 对应的旋转框（可选，用于调试）
#         """
#         self.flow_features.append(flow_feat)
#         self.displacements.append(disp)
#         self.angle_changes.append(angle_delta)
#         self.timestamps.append(frame_id)
#         if bbox is not None:
#             self.bboxes.append(bbox)
#         # 保持队列长度
#         if len(self.flow_features) > self.max_length:
#             self.flow_features.pop(0)
#             self.displacements.pop(0)
#             self.angle_changes.pop(0)
#             self.timestamps.pop(0)
#             if self.bboxes:
#                 self.bboxes.pop(0)

#     def compute_consistency(self, curr_flow_feat, curr_disp, curr_angle_delta, curr_frame_id):
#         """
#         计算当前观测与历史记忆的一致性得分，范围 [0,1]，越高越一致
#         """
#         if len(self.flow_features) < 3:   # 历史不足，返回中性值
#             return 0.5

#         # 时间衰减权重（越近的帧权重越高）
#         time_diffs = np.array([curr_frame_id - t for t in self.timestamps])
#         time_weights = np.exp(-time_diffs / 10.0)   # 衰减因子，可调
#         time_weights /= time_weights.sum() + 1e-6

#         # 计算各特征的相似度
#         flow_sims = []
#         disp_sims = []
#         angle_sims = []
#         for i in range(len(self.flow_features)):
#             # 光流特征余弦相似度
#             f1 = self.flow_features[i]
#             f2 = curr_flow_feat
#             if f1 is not None and f2 is not None:
#                 norm1 = np.linalg.norm(f1)
#                 norm2 = np.linalg.norm(f2)
#                 if norm1 > 0 and norm2 > 0:
#                     cos_sim = np.dot(f1, f2) / (norm1 * norm2)
#                     flow_sims.append(max(0, cos_sim))   # 裁剪到[0,1]
#                 else:
#                     flow_sims.append(0.5)
#             else:
#                 flow_sims.append(0.5)

#             # 位移相似度（欧氏距离归一化）
#             d1 = self.displacements[i]
#             d2 = curr_disp
#             disp_dist = np.linalg.norm(np.array(d1) - np.array(d2))
#             # 假设最大位移为20像素（可调），将距离映射到[0,1]相似度
#             disp_sim = max(0, 1 - disp_dist / 20.0)
#             disp_sims.append(disp_sim)

#             # 角度变化相似度
#             a1 = self.angle_changes[i]
#             a2 = curr_angle_delta
#             angle_diff = abs(a1 - a2)
#             angle_sim = max(0, 1 - angle_diff / (np.pi/4))   # 假设最大角度差为45度
#             angle_sims.append(angle_sim)

#         # 加权融合
#         flow_score = np.average(flow_sims, weights=time_weights[:len(flow_sims)])
#         disp_score = np.average(disp_sims, weights=time_weights[:len(disp_sims)])
#         angle_score = np.average(angle_sims, weights=time_weights[:len(angle_sims)])

#         # 综合得分（可调整权重）
#         consistency = 0.3 * flow_score + 0.4 * disp_score + 0.3 * angle_score
#         return min(max(consistency, 0.0), 1.0)

#     def get_recent_motion(self, num=3):
#         """获取最近num帧的平均运动特征（用于预测或调试）"""
#         if len(self.flow_features) < num:
#             return None
#         recent_flows = np.array(self.flow_features[-num:])
#         recent_disps = np.array(self.displacements[-num:])
#         recent_angles = np.array(self.angle_changes[-num:])
#         return {
#             'mean_flow': np.mean(recent_flows, axis=0),
#             'mean_disp': np.mean(recent_disps, axis=0),
#             'mean_angle_delta': np.mean(recent_angles)
#         }


# def extract_motion_features(prev_gray, curr_gray, prev_bbox, curr_bbox, frame_id):
#     """
#     从连续两帧和对应旋转框中提取运动特征
#     返回: (flow_feat, displacement, angle_delta) 或 (None, None, None) 若失败
#     """
#     if prev_gray is None or curr_gray is None:
#         return None, None, None
#     if prev_gray.shape != curr_gray.shape:
#         return None, None, None

#     # 提取 prev_bbox 区域掩码
#     mask = np.zeros_like(prev_gray, dtype=np.uint8)
#     points = cv2.boxPoints(((prev_bbox[0], prev_bbox[1]),
#                             (prev_bbox[2], prev_bbox[3]),
#                             np.degrees(prev_bbox[4])))
#     points = np.int32(points)
#     cv2.fillPoly(mask, [points], 255)

#     # 计算稠密光流
#     flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
#                                         0.5, 3, 15, 3, 5, 1.2, 0)
#     # 提取目标区域内的光流
#     roi_flow = flow[mask == 255]
#     if len(roi_flow) < 10:
#         return None, None, None

#     # 光流幅度和角度
#     flow_mag = np.sqrt(roi_flow[:, 0]**2 + roi_flow[:, 1]**2)
#     flow_angle = np.arctan2(roi_flow[:, 1], roi_flow[:, 0])

#     # 构建光流特征：幅度直方图(8bin) + 角度直方图(8bin) + 统计量
#     mag_hist, _ = np.histogram(flow_mag, bins=8, range=(0, 20))
#     angle_hist, _ = np.histogram(flow_angle, bins=8, range=(-np.pi, np.pi))
#     # 归一化直方图
#     mag_hist = mag_hist / (len(roi_flow) + 1e-6)
#     angle_hist = angle_hist / (len(roi_flow) + 1e-6)
#     stats = np.array([np.mean(flow_mag), np.std(flow_mag),
#                       np.mean(flow_angle), np.std(flow_angle)])
#     flow_feat = np.concatenate([mag_hist, angle_hist, stats])

#     # 计算位移和角度变化
#     prev_center = np.array([prev_bbox[0], prev_bbox[1]])
#     curr_center = np.array([curr_bbox[0], curr_bbox[1]])
#     displacement = curr_center - prev_center
#     angle_delta = curr_bbox[4] - prev_bbox[4]   # 注意角度周期性，但仔猪角度变化小，暂不处理wrap

#     return flow_feat, displacement, angle_delta

# vim: expandtab:ts=4:sw=4


# vim: expandtab:ts=4:sw=4
"""
运动记忆增强模块
为每个轨迹维护运动模式记忆库，用于计算当前观测与历史运动模式的一致性
完全无需训练，仅使用经典计算机视觉方法
"""
import cv2
import numpy as np

class MotionMemory:
    """
    运动模式记忆库
    存储轨迹的历史运动特征，并提供一致性计算方法
    """
    def __init__(self, max_length=10):
        self.max_length = max_length          # 记忆库最大长度
        self.flow_features = []                # 光流特征列表
        self.displacements = []                 # 位移向量列表 (dx, dy)
        self.angle_changes = []                 # 角度变化列表
        self.timestamps = []                     # 对应帧号
        self.bboxes = []                          # 对应检测框（可选）

    def add(self, flow_feat, disp, angle_delta, frame_id, bbox=None):
        """添加新的运动观测到记忆库"""
        self.flow_features.append(flow_feat)
        self.displacements.append(disp)
        self.angle_changes.append(angle_delta)
        self.timestamps.append(frame_id)
        if bbox is not None:
            self.bboxes.append(bbox)
        # 保持队列长度
        if len(self.flow_features) > self.max_length:
            self.flow_features.pop(0)
            self.displacements.pop(0)
            self.angle_changes.pop(0)
            self.timestamps.pop(0)
            if self.bboxes:
                self.bboxes.pop(0)

    def compute_consistency(self, curr_flow_feat, curr_disp, curr_angle_delta, curr_frame_id):
        """计算当前观测与历史记忆的一致性得分"""
        if len(self.flow_features) < 3:
            return 0.5

        # 时间衰减权重
        time_diffs = np.array([curr_frame_id - t for t in self.timestamps])
        time_weights = np.exp(-time_diffs / 10.0)
        time_weights /= time_weights.sum() + 1e-6

        flow_sims = []
        disp_sims = []
        angle_sims = []
        for i in range(len(self.flow_features)):
            # 光流特征余弦相似度
            f1 = self.flow_features[i]
            f2 = curr_flow_feat
            if f1 is not None and f2 is not None:
                norm1 = np.linalg.norm(f1)
                norm2 = np.linalg.norm(f2)
                if norm1 > 0 and norm2 > 0:
                    cos_sim = np.dot(f1, f2) / (norm1 * norm2)
                    flow_sims.append(max(0, cos_sim))
                else:
                    flow_sims.append(0.5)
            else:
                flow_sims.append(0.5)

            # 位移相似度
            d1 = self.displacements[i]
            d2 = curr_disp
            disp_dist = np.linalg.norm(np.array(d1) - np.array(d2))
            disp_sim = max(0, 1 - disp_dist / 20.0)
            disp_sims.append(disp_sim)

            # 角度变化相似度
            a1 = self.angle_changes[i]
            a2 = curr_angle_delta
            angle_diff = abs(a1 - a2)
            angle_sim = max(0, 1 - angle_diff / (np.pi/4))
            angle_sims.append(angle_sim)

        # 加权融合
        flow_score = np.average(flow_sims, weights=time_weights[:len(flow_sims)])
        disp_score = np.average(disp_sims, weights=time_weights[:len(disp_sims)])
        angle_score = np.average(angle_sims, weights=time_weights[:len(angle_sims)])

        consistency = 0.3 * flow_score + 0.4 * disp_score + 0.3 * angle_score
        return min(max(consistency, 0.0), 1.0)

    def get_recent_motion(self, num=3):
        if len(self.flow_features) < num:
            return None
        recent_flows = np.array(self.flow_features[-num:])
        recent_disps = np.array(self.displacements[-num:])
        recent_angles = np.array(self.angle_changes[-num:])
        return {
            'mean_flow': np.mean(recent_flows, axis=0),
            'mean_disp': np.mean(recent_disps, axis=0),
            'mean_angle_delta': np.mean(recent_angles)
        }


def extract_motion_features(prev_gray, curr_gray, prev_bbox, curr_bbox, frame_id):
    """
    轻量化优化版：带极端宽高防护的安全光流提取
    """
    if prev_gray is None or curr_gray is None:
        return None, None, None
    if prev_gray.shape != curr_gray.shape:
        return None, None, None

    cx, cy, w, h, angle = prev_bbox[:5]
    
    # 预防检测框出现负数宽高的异常
    w, h = abs(w), abs(h)
    
    pad = 20
    min_x = max(0, int(cx - w/2 - pad))
    max_x = min(prev_gray.shape[1], int(cx + w/2 + pad))
    min_y = max(0, int(cy - h/2 - pad))
    max_y = min(prev_gray.shape[0], int(cy + h/2 + pad))

    if max_x <= min_x or max_y <= min_y:
        return None, None, None

    prev_roi = prev_gray[min_y:max_y, min_x:max_x]
    curr_roi = curr_gray[min_y:max_y, min_x:max_x]
    
    h_roi, w_roi = prev_roi.shape[:2]
    
    # 如果极度异常（几乎是一条线），直接跳过计算
    if h_roi == 0 or w_roi == 0:
        return None, None, None

    max_dim = 80
    if max(h_roi, w_roi) > max_dim:
        scale = max_dim / float(max(h_roi, w_roi))
        # 核心防护：强制长宽最小为 1，防止缩放后变为 0 像素报错
        new_w = max(1, int(w_roi * scale))
        new_h = max(1, int(h_roi * scale))
        prev_roi = cv2.resize(prev_roi, (new_w, new_h))
        curr_roi = cv2.resize(curr_roi, (new_w, new_h))

    flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    
    flow_mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    flow_angle = np.arctan2(flow[..., 1], flow[..., 0])

    mag_hist, _ = np.histogram(flow_mag, bins=8, range=(0, 20))
    angle_hist, _ = np.histogram(flow_angle, bins=8, range=(-np.pi, np.pi))
    
    num_pixels = flow_mag.size + 1e-6
    mag_hist = mag_hist / num_pixels
    angle_hist = angle_hist / num_pixels
    stats = np.array([np.mean(flow_mag), np.std(flow_mag),
                      np.mean(flow_angle), np.std(flow_angle)])
    flow_feat = np.concatenate([mag_hist, angle_hist, stats])

    prev_center = np.array([prev_bbox[0], prev_bbox[1]])
    curr_center = np.array([curr_bbox[0], curr_bbox[1]])
    displacement = curr_center - prev_center
    angle_delta = curr_bbox[4] - prev_bbox[4]

    return flow_feat, displacement, angle_delta