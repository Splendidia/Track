# vim: expandtab:ts=4:sw=4
"""
频域角度校准模块 (Fourier Angle Calibration, FAC)
利用二维快速傅里叶变换 (FFT) 提取目标在频域的能量峰值，
计算高置信度的物理真实角度，用于在线修正检测器的噪声角度。
"""
import cv2
import numpy as np

class FourierAngleCalibrator:
    def __init__(self, patch_size=64, angle_bins=180):
        self.patch_size = patch_size
        self.angle_bins = angle_bins
        # 汉宁窗：消除图像边缘截断带来的高频十字星伪影噪声
        self.hanning = cv2.createHanningWindow((patch_size, patch_size), cv2.CV_32F)
        
    def extract_patch(self, img, cx, cy, w, h):
        """截取包含目标的方形图像块并处理边界"""
        size = int(max(w, h) * 1.2)  # 稍微放大一点包含完整边缘
        if size == 0: return None
        
        x1, y1 = int(cx - size / 2), int(cy - size / 2)
        x2, y2 = x1 + size, y1 + size
        img_h, img_w = img.shape[:2]
        
        # 处理越界填充 (Padding)
        pad_left, pad_top = max(0, -x1), max(0, -y1)
        pad_right, pad_bottom = max(0, x2 - img_w), max(0, y2 - img_h)
        
        x1_v, y1_v = max(0, x1), max(0, y1)
        x2_v, y2_v = min(img_w, x2), min(img_h, y2)
        
        patch = img[y1_v:y2_v, x1_v:x2_v]
        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            
        patch = cv2.resize(patch, (self.patch_size, self.patch_size))
        return patch

    def calibrate(self, img, rbox, conf_thresh=1.5):
        """
        传入当前灰度图和检测框，返回校准后的角度和置信度
        """
        cx, cy, w, h, orig_angle = rbox[:5]
        patch = self.extract_patch(img, cx, cy, w, h)
        if patch is None:
            return orig_angle, 0.0
            
        # 1. 预处理与 FFT 变换
        patch_f = np.float32(patch)
        patch_f = patch_f - np.mean(patch_f)  # 去直流分量 (去中心化)
        patch_f = patch_f * self.hanning       # 加窗
        
        f = np.fft.fft2(patch_f)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)                  # 获取振幅谱
        
        # 2. 极坐标转换 (warpPolar) 沿着角度积分
        center = (self.patch_size // 2, self.patch_size // 2)
        max_r = self.patch_size // 2
        # warpPolar 把圆盘拉平成长方形，y轴是角度，x轴是半径
        polar = cv2.warpPolar(mag, (max_r, self.angle_bins), center, max_r, cv2.WARP_POLAR_LINEAR)
        
        # 沿着半径求和，得到每个角度上的总能量（忽略最中心的极低频噪声）
        energy = np.sum(polar[:, 3:], axis=1)
        
        # 频域对称性，只看 0~180 度 (合并前后半段能量)
        half = self.angle_bins // 2
        energy_half = energy[:half] + energy[half:]
        
        # 3. 寻找能量最高的主方向
        best_idx = np.argmax(energy_half)
        peak_energy = energy_half[best_idx]
        mean_energy = np.mean(energy_half)
        
        # 能量集中度即为置信度
        confidence = peak_energy / (mean_energy + 1e-6)
        
        # 如果谱图太乱（比如圆形物体或严重遮挡），直接弃用，返回原角度
        if confidence < conf_thresh:
            return orig_angle, confidence
            
        # 4. 计算校准角度 (频域能量方向与空间物理方向正交)
        freq_angle_deg = best_idx * (360.0 / self.angle_bins) 
        spatial_angle_deg = (freq_angle_deg + 90.0) % 180.0
        spatial_angle_rad = np.radians(spatial_angle_deg)
        
        # 解决 180 度翻转歧义：让它和检测器的原角度靠得最近
        diff1 = abs(orig_angle - spatial_angle_rad)
        diff2 = abs(orig_angle - (spatial_angle_rad + np.pi))
        diff3 = abs(orig_angle - (spatial_angle_rad - np.pi))
        best_rad = [spatial_angle_rad, spatial_angle_rad + np.pi, spatial_angle_rad - np.pi][np.argmin([diff1, diff2, diff3])]
        
        return best_rad, confidence