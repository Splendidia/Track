# vim: expandtab:ts=4:sw=4
"""
遮挡感知运动建模模块 (Occlusion-Aware Motion Modeling, OAMM)
利用纯几何线索（面积、长宽比）和指数移动平均（EMA）评估目标遮挡状态，
用于动态调整匹配阈值并保护外观模板免受污染。
"""
import numpy as np

class OAMM:
    def __init__(self, ema_alpha=0.2, occlusion_thresh=0.4):
        """
        :param ema_alpha: EMA的更新率，越小历史权重越大
        :param occlusion_thresh: 判定为遮挡的分数阈值
        """
        self.ema_alpha = ema_alpha
        self.occlusion_thresh = occlusion_thresh
        
        # 正常状态的参考值
        self.ref_area = None
        self.ref_aspect_ratio = None
        
        # 当前状态
        self.current_score = 0.0
        self.is_occluded = False

    def update_and_score(self, rbox):
        """
        输入当前检测框或预测框，计算遮挡得分，并动态更新正常参考值
        rbox: [x, y, w, h, angle]
        返回: occlusion_score (0.0 ~ 1.0)
        """
        # 提取宽高，加入微小值防止除以0
        w = max(1e-3, abs(rbox[2]))
        h = max(1e-3, abs(rbox[3]))
        
        current_area = w * h
        # 统一长宽比计算方式：长边 / 短边 (保证 >= 1.0)
        current_ar = max(w, h) / min(w, h)

        # 初始帧，直接记录为参考标准
        if self.ref_area is None:
            self.ref_area = current_area
            self.ref_aspect_ratio = current_ar
            self.current_score = 0.0
            self.is_occluded = False
            return 0.0

        # --- 计算几何形变偏差 ---
        # 面积偏差比：如果是两只猪挤在一起，面积可能翻倍；如果是被挡住一半，面积减半
        area_ratio = current_area / self.ref_area
        area_dev = max(area_ratio, 1.0 / area_ratio) - 1.0  # 0代表完美一致

        # 长宽比偏差：猪被遮挡时，框的形状通常会剧烈拉长或变方
        ar_ratio = current_ar / self.ref_aspect_ratio
        ar_dev = max(ar_ratio, 1.0 / ar_ratio) - 1.0

        # --- 计算综合遮挡得分 (可调权重) ---
        # 面积变化权重占 60%，长宽比变化占 40%
        raw_score = (area_dev * 0.6) + (ar_dev * 0.4)
        
        # 将得分映射并截断到 [0, 1] 之间 (假设偏差 1.5 已经是非常严重的遮挡)
        self.current_score = min(1.0, raw_score / 1.5)
        
        # 判断是否触发遮挡警报
        self.is_occluded = self.current_score > self.occlusion_thresh

        # --- 核心保护机制：只在“正常状态”下更新参考值 ---
        # 如果正在发生遮挡，拒绝把畸变的面积更新到历史参考中！
        if not self.is_occluded:
            self.ref_area = (1 - self.ema_alpha) * self.ref_area + self.ema_alpha * current_area
            self.ref_aspect_ratio = (1 - self.ema_alpha) * self.ref_aspect_ratio + self.ema_alpha * current_ar

        return self.current_score