# vim: expandtab:ts=4:sw=4
import cv2
import numpy as np
from tools.track_test.kalman_filter_rbox import KalmanFilter_Rbox, KalmanFilter8D
from tools.track_test.imm_kalman import IMMFilter
from tools.track_test.motion_signature import MotionSignatureExtractor
from tools.track_test.motion_memory import MotionMemory, extract_motion_features
from tools.track_test.sgmp import SGMP  # 🌟 仅引入纯正的 SGMP

class TrackState:
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    def __init__(self, rbox, track_id, conf, class_id, n_init, max_age,
                 use_10d_kf=False, use_imm=False, use_signature=False,
                 use_motion_memory=False, use_oamm=False, use_sgmp=False, use_rifm=False, n_sig_components=5): # 🌟 更改为 use_sgmp
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
        self.use_motion_memory = use_motion_memory
        self.use_sgmp = use_sgmp # 正式接收参数
        self.use_oamm = use_oamm
        self.use_rifm = use_rifm
        ##
        self.rifm_memory = []
        ##
        if self.use_imm:
            self.imm_filter = IMMFilter()
            init_meas = rbox[:5] if len(rbox) == 5 else self._rbox8_to_5(rbox)
            self.mean, self.covariance, self.imm_means, self.imm_covs = self.imm_filter.initiate(init_meas)
            self.angle = init_meas[4]
        elif use_10d_kf:
            self.kf = KalmanFilter_Rbox()
            init_meas = rbox[:5] if len(rbox) == 5 else self._rbox8_to_5(rbox)
            self.mean, self.covariance = self.kf.initiate(init_meas)
            self.angle = init_meas[4]
        else:
            self.kf = KalmanFilter8D()
            if len(rbox) == 8:
                pts = np.array(rbox).reshape(4, 2)
                cx, cy = pts.mean(axis=0)
                w = np.linalg.norm(pts[0] - pts[1])
                h = np.linalg.norm(pts[1] - pts[2])
                self.angle = np.arctan2(pts[1][1]-pts[0][1], pts[1][0]-pts[0][0])
                init_meas = [cx, cy, w, h]
            else:
                cx, cy, w, h, self.angle = rbox[:5]
                init_meas = [cx, cy, w, h]
            self.mean, self.covariance = self.kf.initiate(init_meas)

        self.sig_extractor = MotionSignatureExtractor(n_components=n_sig_components) if use_signature else None
        self.sig_initialized = False
        self.motion_memory = MotionMemory(max_length=10) if use_motion_memory else None
        
        self.rifm_feature = None 
        self.last_rbox = rbox
        
        # 🌟 初始化 SGMP 模块
        if self.use_sgmp:
            self.sgmp = SGMP()
            init_5d = rbox[:5] if len(rbox) == 5 else self._rbox8_to_5(rbox)
            self.sgmp.update_history(init_5d)

    def _rbox8_to_5(self, rbox):
        rect = cv2.minAreaRect(np.array(rbox).reshape(4, 2).astype(np.float32))
        return [rect[0][0], rect[0][1], rect[1][0], rect[1][1], np.radians(rect[2])]

    def get_rbox(self):
        # 信任卡尔曼的平滑力量，消灭检测框的高频抖动，保护 IDF1！
        if self.use_10d_kf or self.use_imm:
            rbox = self.mean[:5].copy()
            if hasattr(self, 'smoothed_angle'):
                rbox[4] = 0.45 * rbox[4] + 0.55 * self.smoothed_angle
            self.smoothed_angle = rbox[4]
            return rbox
        return np.r_[self.mean[:4], self.angle]

    def predict(self, kf):
        if self.use_imm:
            self.mean, self.covariance, self.imm_means, self.imm_covs = self.imm_filter.predict(self.imm_means, self.imm_covs)
        else:
            self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
            
        # 🌟 SGMP 强势介入：进行动静分类约束及物理可行域投影
        if getattr(self, 'use_sgmp', False):
            self.mean, self.covariance = self.sgmp.constrain_prediction(self.mean, self.covariance)

        self.time_since_update += 1

    def update(self, rbox, conf, class_id, prev_gray=None, curr_gray=None, frame_id=None):
        self.conf, self.class_id, self.rbox = conf, class_id, rbox
        
        # 🌟 SGMP 数据采集：收集真实的物理运动数据
        if getattr(self, 'use_sgmp', False):
            meas_5d = rbox[:5] if len(rbox) == 5 else self._rbox8_to_5(rbox)
            prev_5d = self.last_rbox[:5] if len(self.last_rbox) == 5 else self._rbox8_to_5(self.last_rbox)
            self.sgmp.update_history(meas_5d, prev_5d)
            
        self.last_rbox = rbox

        # 一旦成功匹配，100% 信任视觉框（置信度拉满）
        meas_conf = 1.0

        if self.use_imm:
            self.mean, self.covariance, self.imm_means, self.imm_covs = self.imm_filter.update(
                self.imm_means, self.imm_covs, rbox[:5] if len(rbox) == 5 else self._rbox8_to_5(rbox))
            self.angle = (rbox[:5] if len(rbox) == 5 else self._rbox8_to_5(rbox))[4]
        elif self.use_10d_kf:
            self.mean, self.covariance = self.kf.update(self.mean, self.covariance, rbox[:5] if len(rbox) == 5 else self._rbox8_to_5(rbox), confidence=meas_conf)
            self.angle = (rbox[:5] if len(rbox) == 5 else self._rbox8_to_5(rbox))[4]
        else:
            if len(rbox) == 8:
                pts = np.array(rbox).reshape(4, 2)
                cx, cy = pts.mean(axis=0)
                w = np.linalg.norm(pts[0] - pts[1])
                h = np.linalg.norm(pts[1] - pts[2])
                self.angle = np.arctan2(pts[1][1]-pts[0][1], pts[1][0]-pts[0][0])
                meas = [cx, cy, w, h]
            else:
                cx, cy, w, h, self.angle = rbox[:5]
                meas = [cx, cy, w, h]
            self.mean, self.covariance = self.kf.update(self.mean, self.covariance, meas, confidence=meas_conf)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative and self.time_since_update > 3: self.state = TrackState.Deleted
        if self.time_since_update > self._max_age: self.state = TrackState.Deleted

    def is_confirmed(self): return self.state == TrackState.Confirmed
    def is_deleted(self): return self.state == TrackState.Deleted
    def is_tentative(self): return self.state == TrackState.Tentative