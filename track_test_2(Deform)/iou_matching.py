# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from tools.track_test import linear_assignment
import shapely
from shapely.geometry import Polygon
import cv2

# ---------- 辅助函数 ----------
def rbox_to_polygon(rbox):
    rbox = np.array(rbox, dtype=np.float32).flatten()
    x, y, w, h, angle = rbox[:5]
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    half_w = w / 2
    half_h = h / 2
    corners = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ])
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated = np.dot(corners, rot_mat.T)
    rotated[:, 0] += x
    rotated[:, 1] += y
    return rotated.flatten()

def ensure_8point(box):
    box = np.array(box, dtype=np.float32).flatten()
    if len(box) == 5:
        return rbox_to_polygon(box)
    else:
        return box.flatten()

def compute_horizontal_iou(box_a, box_b):
    pts_a = ensure_8point(box_a).reshape(4, 2)
    pts_b = ensure_8point(box_b).reshape(4, 2)
    rect_a = cv2.boundingRect(pts_a.astype(np.int32))
    rect_b = cv2.boundingRect(pts_b.astype(np.int32))
    x1 = max(rect_a[0], rect_b[0])
    y1 = max(rect_a[1], rect_b[1])
    x2 = min(rect_a[0] + rect_a[2], rect_b[0] + rect_b[2])
    y2 = min(rect_a[1] + rect_a[3], rect_b[1] + rect_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = rect_a[2] * rect_a[3]
    area_b = rect_b[2] * rect_b[3]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def iou_eight(bbox, candidates):
    a = np.array(bbox).reshape(4, 2)
    b = np.array(candidates).reshape(4, 2)
    poly1 = Polygon(a).convex_hull
    poly2 = Polygon(b).convex_hull
    if not poly1.intersects(poly2):
        return 0.0
    try:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area
    except shapely.geos.TopologicalError:
        return 0.0
# --------------------------------

def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None, use_riou=False,
             use_deform=False, deform_reg=None, track_templates=None,
             prev_gray=None, curr_gray=None, use_signature=False):
    """
    计算IoU成本矩阵
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 3:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        track = tracks[track_idx]
        track_rbox = track.get_rbox()
        track_rbox = np.array(track_rbox, dtype=np.float32)

        col = 0
        for i in detection_indices:
            det_box = detections[i][0]
            det_box = np.array(det_box, dtype=np.float32)

            # 计算IoU
            if use_riou:
                if len(track_rbox) == 5:
                    trk_8 = rbox_to_polygon(track_rbox)
                else:
                    trk_8 = track_rbox.flatten()
                if len(det_box) == 5:
                    det_8 = rbox_to_polygon(det_box)
                else:
                    det_8 = det_box.flatten()
                iou = iou_eight(trk_8, det_8)
            else:
                iou = compute_horizontal_iou(track_rbox, det_box)

            # 可变形配准相似度
            deform_sim = 1.0
            if use_deform and deform_reg and track_templates and prev_gray is not None and curr_gray is not None:
                track_id = tracks[track_idx].track_id
                if track_id in track_templates:
                    template_info = track_templates[track_id]
                    if template_info['age'] <= 5:
                        candidate_patch = deform_reg.extract_patch(curr_gray, det_box)
                        if candidate_patch is not None and template_info['patch'] is not None:
                            deform_sim = deform_reg.compute_similarity(template_info['patch'], candidate_patch)

            # 运动签名距离
            sig_dist = 0.0
            if use_signature and prev_gray is not None and curr_gray is not None:
                if hasattr(track, 'sig_extractor') and track.sig_initialized:
                    raw_feat = track.sig_extractor.extract_raw_features(prev_gray, curr_gray,
                                                                         track_rbox, det_box)
                    if raw_feat is not None:
                        sig_dist = track.sig_extractor.compute_distance(raw_feat)
                        # 调试打印（可选）
                        #print(f"Match: track {track.track_id}, sig_dist={sig_dist:.4f}")
                    else:
                        sig_dist = 1.0
                else:
                    sig_dist = 0.0

            # 融合各项
            alpha = 0.6   # IoU权重
            beta = 0.3    # 配准权重
            gamma = 0.01   # 签名距离权重
            if use_deform:
                combined_sim = alpha * iou + beta * deform_sim
            else:
                combined_sim = iou

            cost = 1.0 - combined_sim + gamma * sig_dist
            cost_matrix[row, col] = min(cost, linear_assignment.INFTY_COST)
            col += 1
    return cost_matrix


def iou_cost_fuse_score(tracks, detections, track_indices=None,
                        detection_indices=None, use_riou=False,
                        use_deform=False, deform_reg=None, track_templates=None,
                        prev_gray=None, curr_gray=None, use_signature=False):
    """
    融合检测分数的IoU成本（用于第一阶段）
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 3:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        track = tracks[track_idx]
        track_rbox = track.get_rbox()
        track_rbox = np.array(track_rbox, dtype=np.float32)

        col = 0
        for i in detection_indices:
            det_box = detections[i][0]
            det_box = np.array(det_box, dtype=np.float32)

            if use_riou:
                if len(track_rbox) == 5:
                    trk_8 = rbox_to_polygon(track_rbox)
                else:
                    trk_8 = track_rbox.flatten()
                if len(det_box) == 5:
                    det_8 = rbox_to_polygon(det_box)
                else:
                    det_8 = det_box.flatten()
                iou = iou_eight(trk_8, det_8)
            else:
                iou = compute_horizontal_iou(track_rbox, det_box)

            deform_sim = 1.0
            if use_deform and deform_reg and track_templates and prev_gray is not None and curr_gray is not None:
                track_id = tracks[track_idx].track_id
                if track_id in track_templates:
                    template_info = track_templates[track_id]
                    if template_info['age'] <= 5:
                        candidate_patch = deform_reg.extract_patch(curr_gray, det_box)
                        if candidate_patch is not None and template_info['patch'] is not None:
                            deform_sim = deform_reg.compute_similarity(template_info['patch'], candidate_patch)

            sig_dist = 0.0
            if 0.3 < iou < 0.7 and use_signature and prev_gray is not None and curr_gray is not None:
                if hasattr(track, 'sig_extractor') and track.sig_initialized:
                    raw_feat = track.sig_extractor.extract_raw_features(prev_gray, curr_gray,
                                                                         track_rbox, det_box)
                    if raw_feat is not None:
                        sig_dist = track.sig_extractor.compute_distance(raw_feat)
                    else:
                        sig_dist = 1.0
                else:
                    sig_dist = 0.0

            alpha = 0.6
            beta = 0.3
            gamma = 0.01
            if use_deform:
                combined_sim = alpha * iou + beta * deform_sim
            else:
                combined_sim = iou

            cost = 1.0 - combined_sim + gamma * sig_dist
            cost_matrix[row, col] = min(cost, linear_assignment.INFTY_COST)
            col += 1

    # 融合检测分数
    iou_sim = 1 - cost_matrix
    det_scores = []
    for det in detections:
        score = det[1]
        if hasattr(score, 'cpu'):
            det_scores.append(np.array(score.cpu()))
        else:
            det_scores.append(np.array(score))
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim