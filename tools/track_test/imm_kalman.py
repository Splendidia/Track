# vim: expandtab:ts=4:sw=4
import numpy as np
from tools.track_test.kalman_filter_rbox import KalmanFilter_Rbox

class IMMFilter:
    """
    交互式多模型 (IMM) 滤波器，用于旋转框跟踪（10维状态）。
    包含三个运动模型，通过调整过程噪声协方差来区分：
       模型1：匀速运动 (CV) - 较小的速度和角速度噪声
       模型2：匀加速 (CA) - 较大的速度噪声（允许速度快速变化）
       模型3：匀速转弯 (CT) - 较大的角速度噪声（允许快速转向）
    模型概率通过似然更新，实现自适应运动模式切换。
    """
    def __init__(self):
        # 基础 Kalman 滤波器，用于参考原始参数
        base_kf = KalmanFilter_Rbox()
        
        # 模型1：匀速 (CV) - 低噪声，相信预测
        kf1 = KalmanFilter_Rbox()
        kf1._std_weight_velocity = base_kf._std_weight_velocity * 0.5   # 速度噪声减半
        kf1._std_weight_angle_vel = base_kf._std_weight_angle_vel * 0.5

        # 模型2：匀加速 (CA) - 高速度噪声，允许速度快速变化
        kf2 = KalmanFilter_Rbox()
        kf2._std_weight_velocity = base_kf._std_weight_velocity * 2.0   # 速度噪声加倍
        kf2._std_weight_angle_vel = base_kf._std_weight_angle_vel * 1.0

        # 模型3：匀速转弯 (CT) - 高角速度噪声，允许转向
        kf3 = KalmanFilter_Rbox()
        kf3._std_weight_velocity = base_kf._std_weight_velocity * 1.0
        kf3._std_weight_angle_vel = base_kf._std_weight_angle_vel * 2.0   # 角速度噪声加倍

        self.filters = [kf1, kf2, kf3]
        
        # 模型概率初始相等
        self.mu = np.array([1/3, 1/3, 1/3])
        
        # 状态转移概率矩阵 (Markov链)，可根据目标机动性调整
        self.P = np.array([[0.95, 0.025, 0.025],
                           [0.025, 0.95, 0.025],
                           [0.025, 0.025, 0.95]])

    def initiate(self, measurement):
        """
        用第一个测量初始化所有子滤波器，并初始化均值和协方差
        """
        means, covs = [], []
        for kf in self.filters:
            mean, cov = kf.initiate(measurement)
            means.append(mean)
            covs.append(cov)
        # 综合均值和协方差（加权平均）
        mean = np.average(means, axis=0, weights=self.mu)
        cov = np.average(covs, axis=0, weights=self.mu)
        return mean, cov, means, covs

    def predict(self, means, covs):
        """
        对每个子滤波器进行预测，并混合得到最终预测
        返回：混合后的均值和协方差，以及各子滤波器的预测均值和协方差
        """
        pred_means, pred_covs = [], []
        for i, (kf, mean, cov) in enumerate(zip(self.filters, means, covs)):
            m, c = kf.predict(mean, cov)
            pred_means.append(m)
            pred_covs.append(c)

        # 混合预测（用于匹配）
        mixed_mean = np.average(pred_means, axis=0, weights=self.mu)
        mixed_cov = np.average(pred_covs, axis=0, weights=self.mu)
        return mixed_mean, mixed_cov, pred_means, pred_covs

    def update(self, means, covs, measurement, confidence=0.0):
        """
        用观测更新各子滤波器，并更新模型概率
        返回：更新后的混合均值和协方差，以及各子滤波器更新后的均值和协方差
        """
        # 计算各模型的似然
        likelihood = np.zeros(3)
        new_means, new_covs = [], []
        for i, (kf, mean, cov) in enumerate(zip(self.filters, means, covs)):
            # 投影到测量空间
            proj_mean, proj_cov = kf.project(mean, cov, confidence)
            # 计算测量残差
            if hasattr(measurement, 'cpu'):
                meas = measurement.cpu().numpy()
            else:
                meas = np.array(measurement)
            innovation = meas - proj_mean
            # 计算似然（高斯分布）
            det = np.linalg.det(proj_cov)
            if det <= 0:
                likelihood[i] = 0
            else:
                exponent = -0.5 * innovation.T @ np.linalg.solve(proj_cov, innovation)
                coeff = 1.0 / np.sqrt((2*np.pi)**len(meas) * det)
                likelihood[i] = coeff * np.exp(exponent)
        
        # 更新模型概率
        c = likelihood @ self.mu   # 归一化常数
        if c > 0:
            self.mu = (likelihood * self.mu) / c
        else:
            self.mu = np.array([1/3, 1/3, 1/3])  # 避免除零，重置为均匀

        # 更新各子滤波器
        for i, (kf, mean, cov) in enumerate(zip(self.filters, means, covs)):
            m, c_ = kf.update(mean, cov, measurement, confidence)
            new_means.append(m)
            new_covs.append(c_)

        # 混合最终状态（用于后续跟踪）
        mixed_mean = np.average(new_means, axis=0, weights=self.mu)
        mixed_cov = np.average(new_covs, axis=0, weights=self.mu)
        return mixed_mean, mixed_cov, new_means, new_covs