# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import cv2 
from tools.track_test import kalman_filter_rbox, linear_assignment, iou_matching
from tools.track_test.track_rbox import Track
from tools.track_test.deformable_registration import DeformableRegistration
from tools.track_test.fac import FourierAngleCalibrator
from tools.track_test.rifm import RIFMExtractor
from tools.track_test.fqa_gate import FQAGate  

class Tracker:
    GATING_THRESHOLD = np.sqrt(kalman_filter_rbox.chi2inv95[4])

    def __init__(self, max_iou_distance=0.85, max_age=30, n_init=3,
                 use_riou=False, use_10d_kf=False, use_imm=False,
                 use_deform=False, use_signature=False, use_motion_memory=False,
                 use_oamm=False, use_sgmp=False, use_fac=False, use_rifm=False, n_sig_components=5, use_bev=False):
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.use_riou = use_riou
        self.use_10d_kf = use_10d_kf
        self.use_imm = use_imm
        self.use_deform = use_deform
        self.use_signature = use_signature
        self.use_motion_memory = use_motion_memory
        self.use_oamm = use_oamm
        self.use_sgmp = use_sgmp
        self.use_fac = use_fac
        self.use_rifm = use_rifm
        self.use_bev = use_bev 
        self.n_sig_components = n_sig_components
        
        self.kf = kalman_filter_rbox.KalmanFilter_Rbox()
        self.tracks = []
        self._next_id = 1

        self.fqa_gate = FQAGate() if (use_deform or use_rifm) else None
        
        if use_deform or use_rifm:
            self.patch_extractor = DeformableRegistration()
            self.track_templates = {} if use_deform else None
        else:
            self.patch_extractor = None

        self.rifm_extractor = RIFMExtractor() if use_rifm else None
        self.fac = FourierAngleCalibrator() if use_fac else None
        self.prev_gray = None

    def _initiate_track(self, rbox, conf, class_id, img_gray=None):
        track = Track(rbox, self._next_id, conf, class_id, self.n_init, self.max_age,
                      use_10d_kf=self.use_10d_kf, use_imm=self.use_imm,
                      use_signature=self.use_signature, use_motion_memory=self.use_motion_memory,
                      use_oamm=self.use_oamm, use_sgmp=self.use_sgmp, use_rifm=self.use_rifm, n_sig_components=self.n_sig_components) 
        self.tracks.append(track)
        self._next_id += 1

        if self.fqa_gate and img_gray is not None:
            is_valid, iqe_score = self.fqa_gate.check_quality(img_gray, rbox)
            if is_valid:
                self._update_template(track, img_gray, rbox, iqe_score)

    def _update_template(self, track, img, rbox, iqe_score):
        scaled_img = self.fqa_gate.fast_downsample(img)
        scaled_box = self.fqa_gate.scale_box(rbox)
        patch = self.patch_extractor.extract_patch(scaled_img, scaled_box)
        
        if patch is not None and patch.size > 0 and min(patch.shape[:2]) > 8:
            if self.use_deform and self.track_templates is not None:
                self.track_templates[track.track_id] = {'patch': patch, 'rbox': rbox, 'age': 0, 'iqe': iqe_score}
            if self.use_rifm:
                track.rifm_feature = self.rifm_extractor.extract_features(patch)

    def predict(self):
        for track in self.tracks: track.predict(self.kf)

    def update(self, det, det_second, curr_gray=None, frame_id=None):
        # self.predict()  
        
        if self.use_fac and curr_gray is not None:
            for item in det + det_second:
                rbox = np.array(item[0], dtype=np.float32)
                if len(rbox) == 5:
                    w, h = rbox[2], rbox[3]
                    aspect_ratio = max(w, h) / (min(w, h) + 1e-5)
                    if aspect_ratio < 1.5: 
                        best_rad, conf = self.fac.calibrate(curr_gray, rbox, conf_thresh=1.5)
                        if conf >= 1.5: rbox[4] = best_rad
                item[0] = rbox

        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 🌟 修复处：正确向匹配中心传递 use_sgmp 参数
        kwargs = {
            'use_riou': self.use_riou,
            'use_deform': self.use_deform, 'deform_reg': self.patch_extractor, 'track_templates': getattr(self, 'track_templates', None),
            'prev_gray': self.prev_gray, 'curr_gray': curr_gray,
            'use_signature': self.use_signature, 'use_motion_memory': self.use_motion_memory,
            'use_oamm': self.use_oamm, 'use_sgmp': self.use_sgmp, 'use_rifm': self.use_rifm, 'rifm_extractor': getattr(self, 'rifm_extractor', None),
            'frame_id': frame_id,
            'use_bev': self.use_bev,
            'fqa_gate': self.fqa_gate
        }

        def cost_with_riou(tracks, dets, tidx, didx):
            return iou_matching.iou_cost_fuse_score(tracks, dets, track_indices=tidx, detection_indices=didx, **kwargs)

        def cost_without_fuse(tracks, dets, tidx, didx):
            return iou_matching.iou_cost(tracks, dets, track_indices=tidx, detection_indices=didx, **kwargs)

        matches_a, un_t_a, un_d_a = linear_assignment.min_cost_matching(
            cost_with_riou, self.max_iou_distance, self.tracks, det, confirmed_tracks)

        matches_b, un_t_b, un_d_b = linear_assignment.min_cost_matching(
            cost_without_fuse, 0.5, self.tracks, det_second, un_t_a)

        matches_c, un_t_c, un_d_c = linear_assignment.min_cost_matching(
            cost_without_fuse, 0.7, self.tracks, det, unconfirmed_tracks, un_d_a)

        for matches_list, det_list in [(matches_a, det), (matches_b, det_second), (matches_c, det)]:
            for track_idx, detection_idx in matches_list:
                track = self.tracks[track_idx]
                det_box = det_list[detection_idx][0]


                ##
                # 🌟 新增：判断这头猪当前是否被遮挡 (可见度门控)
                # 计算当前猪与其他所有猪框的重叠率
                # all_other_boxes = [d[0] for d in det if d != det_list[detection_idx]]
                all_other_boxes = [det_list[j][0] for j in range(len(det_list)) if j != detection_idx]
                # 使用 iou_matching 中的工具计算最大重叠
                max_overlap = 0.0
                for other_box in all_other_boxes:
                    overlap = iou_matching.compute_horizontal_iou(det_box, other_box)
                    max_overlap = max(max_overlap, overlap)
        
                # 🌟 只有不重叠且图像清晰(FQA)时，才存入纯净记忆库
                if max_overlap < 0.1 and self.rifm_extractor is not None:
                    if self.fqa_gate is not None:
                        is_valid, _ = self.fqa_gate.check_quality(curr_gray, det_box)
                        if is_valid:
                            patch = self.patch_extractor.extract_patch(curr_gray, det_box)
                            feat = self.rifm_extractor.extract_features(patch)
                            track.rifm_memory.append(feat)
                            if len(track.rifm_memory) > 3: track.rifm_memory.pop(0) # 保持最近3帧

                ##


                det_score = det_list[detection_idx][1]
                det_class = det_list[detection_idx][2]

                track.update(det_box, det_score, det_class, prev_gray=self.prev_gray, curr_gray=curr_gray)

                if self.fqa_gate and curr_gray is not None:
                    needs_update = True
                    if self.track_templates is not None and track.track_id in self.track_templates:
                        if self.track_templates[track.track_id]['age'] < 10: 
                            needs_update = False
                    
                    if needs_update:
                        is_valid, iqe_score = self.fqa_gate.check_quality(curr_gray, det_box)
                        if is_valid: 
                            self._update_template(track, curr_gray, det_box, iqe_score)

        for track_idx in un_t_b + un_t_c:
            self.tracks[track_idx].mark_missed()

        for detection_idx in un_d_c:
            self._initiate_track(det[detection_idx][0], det[detection_idx][1], det[detection_idx][2], curr_gray)

        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        if self.use_deform and self.track_templates is not None:
            current_ids = {t.track_id for t in self.tracks}
            self.track_templates = {tid: tmpl for tid, tmpl in self.track_templates.items() if tid in current_ids}
            for tid in self.track_templates: self.track_templates[tid]['age'] += 1

        if curr_gray is not None:
            self.prev_gray = curr_gray.copy()

    def get_active_tracks(self):
        return [t for t in self.tracks if t.is_confirmed() and t.time_since_update == 0]