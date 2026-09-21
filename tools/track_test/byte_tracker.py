# import numpy as np
# from tools.track_test.tracker import Tracker

# class Byte_tracker(object):
#     def __init__(self, max_iou_distance=0.85, max_age=30, n_init=3,
#                  use_riou=False, use_10d_kf=False, use_imm=False,
#                  use_deform=False, use_signature=False, use_motion_memory=False,
#                  use_oamm=False, use_fac=False, use_rifm=False, n_sig_components=5, use_bev=False): # 新增 use_bev
        
#         self.track_thresh = 0.5
#         self.low_thresh = 0.1

#         self.tracker = Tracker(
#             max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init,
#             use_riou=use_riou, use_10d_kf=use_10d_kf, use_imm=use_imm,
#             use_deform=use_deform, use_signature=use_signature, use_motion_memory=use_motion_memory,
#             use_oamm=use_oamm, use_fac=use_fac, use_rifm=use_rifm, n_sig_components=n_sig_components,
#             use_bev=use_bev # 透传给底层 Tracker
#         )

#     def update(self, boxes, scores, classes, img=None, frame_id=None):
#         det = []
#         det_second = []
#         for box, score, cls_id in zip(boxes, scores, classes):
#             if score >= self.track_thresh:
#                 det.append([box, score, cls_id])
#             elif score >= self.low_thresh:
#                 det_second.append([box, score, cls_id])
                
#         self.tracker.update(det, det_second, curr_gray=img, frame_id=frame_id)
        
#         results = []
#         for track in self.tracker.get_active_tracks():
#             conf = getattr(track, 'score', getattr(track, 'conf', 1.0))
#             results.append([track.get_rbox(), track.track_id, track.class_id, conf])
#         return results
import numpy as np
from tools.track_test.tracker import Tracker

class Byte_tracker(object):
    # def __init__(self, max_iou_distance=0.85, max_age=30, n_init=3,
    #              use_riou=False, use_10d_kf=False, use_imm=False,
    #              use_deform=False, use_signature=False, use_motion_memory=False,
    #              use_sgmp=False, use_fac=False, use_rifm=False, n_sig_components=5, use_bev=False): # 🌟 更改为 use_sgmp
        
    #     self.track_thresh = 0.5
    #     self.low_thresh = 0.1

    #     self.tracker = Tracker(
    #         max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init,
    #         use_riou=use_riou, use_10d_kf=use_10d_kf, use_imm=use_imm,
    #         use_deform=use_deform, use_signature=use_signature, use_motion_memory=use_motion_memory,
    #         use_sgmp=use_sgmp, use_fac=use_fac, use_rifm=use_rifm, n_sig_components=n_sig_components, # 🌟 更改为 use_sgmp
    #         use_bev=use_bev
    #     )
    def __init__(self, max_iou_distance=0.85, max_age=30, n_init=3,
                 use_riou=False, use_10d_kf=False, use_imm=False,
                 use_deform=False, use_signature=False, use_motion_memory=False,
                 use_oamm=False, use_sgmp=False, use_fac=False, use_rifm=False, n_sig_components=5, use_bev=False):
        

        self.track_thresh = 0.5
        self.low_thresh = 0.1

        self.tracker = Tracker(
            max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init,
            use_riou=use_riou, use_10d_kf=use_10d_kf, use_imm=use_imm,
            use_deform=use_deform, use_signature=use_signature, use_motion_memory=use_motion_memory,
            use_oamm=use_oamm, # 🌟 这里也统一叫 use_oamm
            use_sgmp=use_sgmp,
            use_fac=use_fac, use_rifm=use_rifm, n_sig_components=n_sig_components,
            use_bev=use_bev
        )

    def update(self, boxes, scores, classes, img=None, frame_id=None):
        det = []
        det_second = []
        for box, score, cls_id in zip(boxes, scores, classes):
            if score >= self.track_thresh:
                det.append([box, score, cls_id])
            elif score >= self.low_thresh:
                det_second.append([box, score, cls_id])
                
        self.tracker.update(det, det_second, curr_gray=img, frame_id=frame_id)
        
        results = []
        for track in self.tracker.get_active_tracks():
            conf = getattr(track, 'score', getattr(track, 'conf', 1.0))
            results.append([track.get_rbox(), track.track_id, track.class_id, conf])
        return results