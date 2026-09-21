import numpy as np
import torch
import cv2
from tools.track_test.tracker import Tracker

__all__ = ['Byte_tracker']

class Byte_tracker(object):
    def __init__(self, 
                 max_iou_distance=0.95,
                 max_age=50,
                 n_init=3,
                 use_riou=False,
                 use_10d_kf=False,
                 use_imm=False,
                 use_deform=False,
                 use_signature=False,
                 n_sig_components=5):
        self.tracker = Tracker(max_iou_distance=max_iou_distance, 
                               max_age=max_age, n_init=n_init,
                               use_riou=use_riou, use_10d_kf=use_10d_kf,
                               use_imm=use_imm, use_deform=use_deform,
                               use_signature=use_signature,
                               n_sig_components=n_sig_components)

    def update(self, rbox, confidences, classes, ori_img):
        if ori_img is not None:
            self.height, self.width = ori_img.shape[:2]
            gray = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = None

        track_high_thresh = 0.5
        track_low_thresh = 0.1
        det = []
        det_second = []

        for i, conf in enumerate(confidences):
            if conf >= track_high_thresh:
                det.append([rbox[i], confidences[i], classes[i]])
            elif conf > track_low_thresh:
                det_second.append([rbox[i], confidences[i], classes[i]])

        self.tracker.predict()
        self.tracker.update(det, det_second, gray)

        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 3:
                continue
            rbox = track.get_rbox()
            track_id = track.track_id
            class_id = track.class_id
            conf = track.conf
            if hasattr(conf, "cpu"):
                conf = conf.cpu().item() if hasattr(conf, "item") else float(conf.cpu())
            else:
                conf = float(conf)
            if torch.is_tensor(rbox):
                rbox = rbox.cpu().numpy()
            outputs.append([rbox, int(track_id), int(class_id), conf])
        return outputs