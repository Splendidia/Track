# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from tools.track_test import kalman_filter_rbox
from tools.track_test import linear_assignment
from tools.track_test import iou_matching
from tools.track_test.track_rbox import Track
from tools.track_test.deformable_registration import DeformableRegistration

class Tracker:
    GATING_THRESHOLD = np.sqrt(kalman_filter_rbox.chi2inv95[4])

    def __init__(self, max_iou_distance=0.85, max_age=30, n_init=3,
                 use_riou=False, use_10d_kf=False, use_imm=False,
                 use_deform=False, use_signature=False, n_sig_components=5):
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.use_riou = use_riou
        self.use_10d_kf = use_10d_kf
        self.use_imm = use_imm
        self.use_deform = use_deform
        self.use_signature = use_signature
        self.n_sig_components = n_sig_components
        self.kf = kalman_filter_rbox.KalmanFilter_Rbox()
        self.tracks = []
        self._next_id = 1

        # 配准模块
        if use_deform:
            self.deform_reg = DeformableRegistration()
            self.track_templates = {}

        # 用于签名的图像缓存
        self.prev_gray = None

    def _initiate_track(self, rbox, conf, class_id, img_gray=None):
        track = Track(rbox, self._next_id, conf, class_id, self.n_init, self.max_age,
                      use_10d_kf=self.use_10d_kf, use_imm=self.use_imm,
                      use_signature=self.use_signature, n_sig_components=self.n_sig_components)
        self.tracks.append(track)
        self._next_id += 1

        # 初始化模板（如果启用配准）
        if self.use_deform and img_gray is not None:
            patch = self.deform_reg.extract_patch(img_gray, rbox)
            if patch is not None:
                self.track_templates[track.track_id] = {'patch': patch, 'rbox': rbox, 'age': 0}

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, det, det_second, curr_gray=None):
        """
        更新跟踪器
        curr_gray: 当前帧灰度图（用于配准和签名）
        """
        # 多级匹配
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 定义代价函数
        def cost_with_riou(tracks, dets, tidx, didx):
            return iou_matching.iou_cost_fuse_score(
                tracks, dets, tidx, didx, use_riou=self.use_riou,
                use_deform=self.use_deform, deform_reg=self.deform_reg if self.use_deform else None,
                track_templates=self.track_templates if self.use_deform else None,
                prev_gray=self.prev_gray, curr_gray=curr_gray,
                use_signature=self.use_signature)

        def cost_without_fuse(tracks, dets, tidx, didx):
            return iou_matching.iou_cost(
                tracks, dets, tidx, didx, use_riou=self.use_riou,
                use_deform=self.use_deform, deform_reg=self.deform_reg if self.use_deform else None,
                track_templates=self.track_templates if self.use_deform else None,
                prev_gray=self.prev_gray, curr_gray=curr_gray,
                use_signature=self.use_signature)

        # 第一阶段匹配（高分框 + 融合分数）
        matches_a, unmatched_tracks_a, unmatched_detections_a = \
            linear_assignment.min_cost_matching(
                cost_with_riou, self.max_iou_distance, self.tracks,
                det, confirmed_tracks)

        # 更新匹配的轨迹
        for track_idx, detection_idx in matches_a:
            track = self.tracks[track_idx]
            det_box = det[detection_idx][0]
            track.update(det_box, det[detection_idx][1], det[detection_idx][2],
                         prev_gray=self.prev_gray, curr_gray=curr_gray)

            # 更新模板
            if self.use_deform and curr_gray is not None:
                patch = self.deform_reg.extract_patch(curr_gray, det_box)
                if patch is not None:
                    self.track_templates[track.track_id] = {'patch': patch, 'rbox': det_box, 'age': 0}

        # 第二阶段匹配（低分框 + 不带融合的 IoU）
        matches_b, unmatched_tracks_b, unmatched_detections_b = \
            linear_assignment.min_cost_matching(
                cost_without_fuse, 0.5, self.tracks, det_second, unmatched_tracks_a)

        for track_idx, detection_idx in matches_b:
            track = self.tracks[track_idx]
            det_box = det_second[detection_idx][0]
            track.update(det_box, det_second[detection_idx][1], det_second[detection_idx][2],
                         prev_gray=self.prev_gray, curr_gray=curr_gray)
            if self.use_deform and curr_gray is not None:
                patch = self.deform_reg.extract_patch(curr_gray, det_box)
                if patch is not None:
                    self.track_templates[track.track_id] = {'patch': patch, 'rbox': det_box, 'age': 0}

        # 未匹配的轨迹标记丢失
        for track_idx in unmatched_tracks_b:
            self.tracks[track_idx].mark_missed()

        # 第三阶段匹配（未确认轨迹）
        matches_c, unmatched_tracks_c, unmatched_detections_c = \
            linear_assignment.min_cost_matching(
                cost_without_fuse, 0.7, self.tracks, det, unconfirmed_tracks, unmatched_detections_a)

        for track_idx, detection_idx in matches_c:
            track = self.tracks[track_idx]
            det_box = det[detection_idx][0]
            track.update(det_box, det[detection_idx][1], det[detection_idx][2],
                         prev_gray=self.prev_gray, curr_gray=curr_gray)
            if self.use_deform and curr_gray is not None:
                patch = self.deform_reg.extract_patch(curr_gray, det_box)
                if patch is not None:
                    self.track_templates[track.track_id] = {'patch': patch, 'rbox': det_box, 'age': 0}

        for track_idx in unmatched_tracks_c:
            self.tracks[track_idx].mark_missed()

        # 剩余检测初始化新轨迹
        for detection_idx in unmatched_detections_c:
            self._initiate_track(det[detection_idx][0], det[detection_idx][1], det[detection_idx][2], curr_gray)

        # 清理已删除轨迹
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        if self.use_deform:
            current_ids = {t.track_id for t in self.tracks}
            self.track_templates = {tid: tmpl for tid, tmpl in self.track_templates.items() if tid in current_ids}
            for tid in self.track_templates:
                self.track_templates[tid]['age'] += 1

        # 保存当前帧灰度图供下一帧使用
        if curr_gray is not None:
            self.prev_gray = curr_gray.copy()

    def get_active_tracks(self):
        return [t for t in self.tracks if t.is_confirmed() and t.time_since_update <= 3]