# vim: expandtab:ts=4:sw=4
import cv2
import numpy as np
from tools.track_test.kalman_filter_rbox import KalmanFilter_Rbox, KalmanFilter8D
from tools.track_test.imm_kalman import IMMFilter
from tools.track_test.motion_signature import MotionSignatureExtractor

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    def __init__(self, rbox, track_id, conf, class_id, n_init, max_age,
                 use_10d_kf=False, use_imm=False, use_signature=False,
                 n_sig_components=5):
        self.track_id = track_id
        self.class_id = class_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.rbox = rbox
        self.state = TrackState.Tentative
        self.conf = conf
        self._n_init = n_init
        self._max_age = max_age
        self.use_10d_kf = use_10d_kf
        self.use_imm = use_imm and use_10d_kf
        self.use_signature = use_signature

        if self.use_imm:
            self.imm_filter = IMMFilter()
            if len(rbox) == 8:
                points = np.array(rbox).reshape(4, 2)
                rect = cv2.minAreaRect(points.astype(np.float32))
                cx, cy = rect[0]
                w, h = rect[1]
                angle = np.radians(rect[2])
                init_meas = [cx, cy, w, h, angle]
            else:
                init_meas = rbox[:5]
            self.mean, self.covariance, self.imm_means, self.imm_covs = \
                self.imm_filter.initiate(init_meas)
            self.angle = init_meas[4]
        elif use_10d_kf:
            self.kf = KalmanFilter_Rbox()
            if len(rbox) == 8:
                points = np.array(rbox).reshape(4, 2)
                rect = cv2.minAreaRect(points.astype(np.float32))
                cx, cy = rect[0]
                w, h = rect[1]
                angle = np.radians(rect[2])
                init_meas = [cx, cy, w, h, angle]
            else:
                init_meas = rbox[:5]
            self.mean, self.covariance = self.kf.initiate(init_meas)
            self.angle = init_meas[4]
        else:
            self.kf = KalmanFilter8D()
            if len(rbox) == 8:
                points = np.array(rbox).reshape(4, 2)
                cx = points[:,0].mean()
                cy = points[:,1].mean()
                w = np.linalg.norm(points[0] - points[1])
                h = np.linalg.norm(points[1] - points[2])
                vec = points[1] - points[0]
                self.angle = np.arctan2(vec[1], vec[0])
                init_meas = [cx, cy, w, h]
            else:
                cx, cy, w, h, self.angle = rbox[:5]
                init_meas = [cx, cy, w, h]
            self.mean, self.covariance = self.kf.initiate(init_meas)

        # 运动签名相关
        if self.use_signature:
            self.sig_extractor = MotionSignatureExtractor()
            self.sig_initialized = False
            self.last_rbox = rbox   # 保存上一帧的检测框用于签名更新
        else:
            self.sig_extractor = None

    def get_rbox(self):
        if self.use_10d_kf or self.use_imm:
            rbox = self.mean[:5].copy()
            if hasattr(self, 'smoothed_angle'):
                alpha = 0.45
                rbox[4] = alpha * rbox[4] + (1 - alpha) * self.smoothed_angle
            self.smoothed_angle = rbox[4]
            return rbox
        else:
            return np.r_[self.mean[:4], self.angle]

    def to_tlwh(self):
        ret = self.mean[:8].copy()
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def to_polygon(self):
        rbox = self.get_rbox()
        if len(rbox) >= 5:
            cx, cy, w, h, angle = rbox[:5]
            from math import cos, sin
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            corners = np.array([
                [-w/2, -h/2],
                [w/2, -h/2],
                [w/2, h/2],
                [-w/2, h/2]
            ])
            rotation_matrix = np.array([[cos_a, -sin_a],
                                         [sin_a, cos_a]])
            rotated_corners = np.dot(corners, rotation_matrix.T)
            rotated_corners[:, 0] += cx
            rotated_corners[:, 1] += cy
            return rotated_corners.flatten()
        else:
            return self.rbox

    def predict(self, kf=None):
        if self.use_imm:
            self.mean, self.covariance, self.imm_means, self.imm_covs = \
                self.imm_filter.predict(self.imm_means, self.imm_covs)
        else:
            self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def update(self, rbox, conf, class_id, prev_gray=None, curr_gray=None):
        """
        更新轨迹
        prev_gray, curr_gray: 用于运动签名更新
        """
        self.conf = conf
        self.class_id = class_id
        self.rbox = rbox

        # 更新几何状态
        if self.use_imm:
            if len(rbox) == 8:
                points = np.array(rbox).reshape(4, 2)
                rect = cv2.minAreaRect(points.astype(np.float32))
                cx, cy = rect[0]
                w, h = rect[1]
                angle = np.radians(rect[2])
                meas = [cx, cy, w, h, angle]
            else:
                meas = rbox[:5]
            self.mean, self.covariance, self.imm_means, self.imm_covs = \
                self.imm_filter.update(self.imm_means, self.imm_covs, meas, conf)
        elif self.use_10d_kf:
            if len(rbox) == 8:
                points = np.array(rbox).reshape(4, 2)
                rect = cv2.minAreaRect(points.astype(np.float32))
                cx, cy = rect[0]
                w, h = rect[1]
                angle = np.radians(rect[2])
                meas = [cx, cy, w, h, angle]
            else:
                meas = rbox[:5]
            self.mean, self.covariance = self.kf.update(self.mean, self.covariance, meas, conf)
        else:
            if len(rbox) == 8:
                points = np.array(rbox).reshape(4, 2)
                cx = points[:,0].mean()
                cy = points[:,1].mean()
                w = np.linalg.norm(points[0] - points[1])
                h = np.linalg.norm(points[1] - points[2])
                vec = points[1] - points[0]
                self.angle = np.arctan2(vec[1], vec[0])
                meas = [cx, cy, w, h]
            else:
                cx, cy, w, h, self.angle = rbox[:5]
                meas = [cx, cy, w, h]
            self.mean, self.covariance = self.kf.update(self.mean, self.covariance, meas, conf)

        # 更新运动签名（如果启用且提供了前后帧图像）
        if self.use_signature and prev_gray is not None and curr_gray is not None:
            raw_feat = self.sig_extractor.extract_raw_features(prev_gray, curr_gray,
                                                                self.last_rbox, rbox)
            if raw_feat is not None:
                self.sig_extractor.update_signature(raw_feat)
                # 为了方便在匹配时使用，将当前签名保存到 self.signature（可选）
                self.signature = raw_feat.copy()
                self.sig_initialized = True
                # print(f"Track {self.track_id}: signature updated")
            else:
                # print(f"Track {self.track_id}: raw_feat is None")
                pass
            self.last_rbox = rbox

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative and self.time_since_update > 3:
            self.state = TrackState.Deleted
        if self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        return self.state == TrackState.Deleted