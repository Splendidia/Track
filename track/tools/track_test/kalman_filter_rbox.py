# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}

class KalmanFilter_Rbox(object):
    def __init__(self):
        ndim, dt = 5, 1.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 200
        # 新增：角度专用权重（可适当调整）
        self._std_weight_angle = 1. / 60
        self._std_weight_angle_vel = 1. / 200

    def initiate(self, measurement):
        if hasattr(measurement, 'cpu'):
            measurement = measurement.cpu().numpy()
        else:
            measurement = np.array(measurement)
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        std = [
            2 * self._std_weight_position * measurement[0],
            2 * self._std_weight_position * measurement[1],
            1 * (measurement[2]/measurement[3]),
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_angle * measurement[4],        # 角度位置
            10 * self._std_weight_velocity * measurement[0],
            10 * self._std_weight_velocity * measurement[1],
            0.1 * (measurement[2]/measurement[3]),
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_angle_vel * measurement[4]    # 角度速度
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[0],
            self._std_weight_position * mean[1],
            1 * (mean[2]/ mean[3]),
            self._std_weight_position * mean[3],
            self._std_weight_angle * mean[4]                    # 角度位置
        ]
        std_vel = [
            self._std_weight_velocity * mean[0],
            self._std_weight_velocity * mean[1],
            0.1 * (mean[2]/ mean[3]),
            self._std_weight_velocity * mean[3],
            self._std_weight_angle_vel * mean[4]                # 角度速度
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance, confidence=.0):
        # 测量噪声，角度项使用专用权重
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
            (self._std_weight_angle * mean[4] + 1e-3)                    # 角度测量噪声
        ]
        if confidence > 0:
            std = [x / (confidence + self._std_weight_position * mean[3]) for x in std]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement, confidence=.0):
        projected_mean, projected_cov = self.project(mean, covariance, confidence)
        # 正则化，防止矩阵奇异
        projected_cov += 1e-6 * np.eye(projected_cov.shape[0])

        if hasattr(measurement, 'cpu'):
            measurement = measurement.cpu().numpy()
        else:
            measurement = np.array(measurement)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class KalmanFilter8D(object):
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        if hasattr(measurement, 'cpu'):
            measurement = measurement.cpu().numpy()
        else:
            measurement = np.array(measurement)
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[0],
            2 * self._std_weight_position * measurement[1],
            1 * (measurement[2] / measurement[3]),
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[0],
            10 * self._std_weight_velocity * measurement[1],
            0.1 * (measurement[2] / measurement[3]),
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[0],
            self._std_weight_position * mean[1],
            1 * (mean[2] / mean[3]),
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[0],
            self._std_weight_velocity * mean[1],
            0.1 * (mean[2] / mean[3]),
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        return mean, covariance

    def project(self, mean, covariance, confidence=0.0):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        if confidence > 0:
            std = [x / (confidence + 1e-6) for x in std]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement, confidence=0.0):
        projected_mean, projected_cov = self.project(mean, covariance, confidence)
        # 正则化
        projected_cov += 1e-6 * np.eye(projected_cov.shape[0])

        if hasattr(measurement, 'cpu'):
            measurement = measurement.cpu().numpy()
        else:
            measurement = np.array(measurement)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance