# # vim: expandtab:ts=4:sw=4
# from __future__ import absolute_import
# import numpy as np
# from tools.track_test import linear_assignment
# import shapely
# from shapely.geometry import Polygon
# import cv2
# from tools.track_test.bev_spatial import BEVSpatial

# def rbox_to_polygon(rbox):
#     rbox = np.array(rbox, dtype=np.float32).flatten()
#     x, y, w, h, angle = rbox[:5]
#     cos_a = np.cos(angle)
#     sin_a = np.sin(angle)
#     half_w = w / 2
#     half_h = h / 2
#     corners = np.array([
#         [-half_w, -half_h],
#         [half_w, -half_h],
#         [half_w, half_h],
#         [-half_w, half_h]
#     ])
#     rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
#     rotated = np.dot(corners, rot_mat.T)
#     rotated[:, 0] += x
#     rotated[:, 1] += y
#     return rotated.flatten()

# def ensure_8point(box):
#     box = np.array(box, dtype=np.float32).flatten()
#     if len(box) == 5:
#         return rbox_to_polygon(box)
#     else:
#         return box.flatten()

# def compute_horizontal_iou(box_a, box_b):
#     pts_a = ensure_8point(box_a).reshape(4, 2)
#     pts_b = ensure_8point(box_b).reshape(4, 2)
#     rect_a = cv2.boundingRect(pts_a.astype(np.int32))
#     rect_b = cv2.boundingRect(pts_b.astype(np.int32))
#     x1 = max(rect_a[0], rect_b[0])
#     y1 = max(rect_a[1], rect_b[1])
#     x2 = min(rect_a[0] + rect_a[2], rect_b[0] + rect_b[2])
#     y2 = min(rect_a[1] + rect_a[3], rect_b[1] + rect_b[3])
#     inter = max(0, x2 - x1) * max(0, y2 - y1)
#     area_a = rect_a[2] * rect_a[3]
#     area_b = rect_b[2] * rect_b[3]
#     union = area_a + area_b - inter
#     return inter / union if union > 0 else 0

# def iou_eight(bbox, candidates):
#     a = np.array(bbox).reshape(4, 2)
#     b = np.array(candidates).reshape(4, 2)
#     poly1 = Polygon(a).convex_hull
#     poly2 = Polygon(b).convex_hull
#     if not poly1.intersects(poly2):
#         return 0.0
#     try:
#         inter_area = poly1.intersection(poly2).area
#         union_area = poly1.area + poly2.area - inter_area
#         if union_area == 0:
#             return 0.0
#         return inter_area / union_area
#     except shapely.geos.TopologicalError:
#         return 0.0

# def iou_cost(tracks, detections, track_indices=None,
#              detection_indices=None, use_riou=False,
#              use_deform=False, deform_reg=None, track_templates=None,
#              prev_gray=None, curr_gray=None, use_signature=False,
#              use_motion_memory=False, use_oamm=False, use_rifm=False, 
#              rifm_extractor=None, frame_id=None, use_bev=False, fqa_gate=None): 
    
#     if track_indices is None:
#         track_indices = np.arange(len(tracks))
#     if detection_indices is None:
#         detection_indices = np.arange(len(detections))

#     cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
#     det_rifm_cache = {} 

#     for row, track_idx in enumerate(track_indices):
#         if tracks[track_idx].time_since_update > 60:
#             cost_matrix[row, :] = linear_assignment.INFTY_COST
#             continue

#         track = tracks[track_idx]
#         track_rbox = track.get_rbox()
#         track_rbox = np.array(track_rbox, dtype=np.float32)
#         is_occluded = use_oamm and getattr(track, 'occlusion_score', 0.0) > 0.08

#         col = 0
#         for i in detection_indices:
#             det_box = detections[i][0]
#             det_box = np.array(det_box, dtype=np.float32)

#             if use_riou:
#                 trk_8 = rbox_to_polygon(track_rbox) if len(track_rbox) == 5 else track_rbox.flatten()
#                 det_8 = rbox_to_polygon(det_box) if len(det_box) == 5 else det_box.flatten()
#                 iou = iou_eight(trk_8, det_8)
#             else:
#                 iou = compute_horizontal_iou(track_rbox, det_box)

#             if use_bev:
#                 bev_sim = BEVSpatial.compute_bev_similarity(track_rbox, det_box)
#                 geom_sim = iou + bev_sim - (iou * bev_sim)
#             else:
#                 geom_sim = iou

#             deform_sim = 1.0
#             if use_deform and deform_reg and track_templates and prev_gray is not None and curr_gray is not None:
#                 if 0.4 < geom_sim < 0.6 and not is_occluded:
#                     track_id = tracks[track_idx].track_id
#                     if track_id in track_templates:
#                         template_info = track_templates[track_id]
#                         if template_info['age'] <= 5:
#                             if fqa_gate:
#                                 candidate_patch = deform_reg.extract_patch(fqa_gate.fast_downsample(curr_gray), fqa_gate.scale_box(det_box))
#                             else:
#                                 candidate_patch = deform_reg.extract_patch(curr_gray, det_box)
                                
#                             if candidate_patch is not None and template_info['patch'] is not None:
#                                 sim = deform_reg.compute_similarity(template_info['patch'], candidate_patch)
#                                 deform_sim = sim if sim > 0.8 else 1.0

#             rifm_sim = 0.0
#             if use_rifm and track.rifm_feature is not None and curr_gray is not None and deform_reg is not None:
#                 if 0.15 < geom_sim < 0.55:  
#                     if i not in det_rifm_cache:
#                         if fqa_gate:
#                             det_patch = deform_reg.extract_patch(fqa_gate.fast_downsample(curr_gray), fqa_gate.scale_box(det_box))
#                         else:
#                             det_patch = deform_reg.extract_patch(curr_gray, det_box)
                            
#                         if det_patch is not None and rifm_extractor is not None:
#                             det_rifm_cache[i] = rifm_extractor.extract_features(det_patch)
#                         else:
#                             det_rifm_cache[i] = None
                    
#                     det_feat = det_rifm_cache[i]
#                     if det_feat is not None:
#                         sim = rifm_extractor.compute_similarity(track.rifm_feature, det_feat)
#                         if sim > 0.75:
#                             rifm_sim = sim

#             occ_score = getattr(track, 'occlusion_score', 0.0) if use_oamm else 0.0

#             if use_deform and use_rifm:
#                 if rifm_sim > 0:
#                     w_vis = max(0.0, 0.5 - occ_score) 
#                     w_deform = w_vis * (0.2 / 0.5)
#                     w_rifm = w_vis * (0.3 / 0.5)
#                     w_geom = 1.0 - w_vis
#                     combined_sim = w_geom * geom_sim + w_deform * deform_sim + w_rifm * rifm_sim
#                 else:
#                     w_deform = max(0.0, 0.3 - occ_score)
#                     w_geom = 1.0 - w_deform
#                     combined_sim = w_geom * geom_sim + w_deform * deform_sim
#             elif use_deform:
#                 decay_factor = np.exp(-1.5 * occ_score)
#                 w_deform = 0.4 * decay_factor
#                 w_geom = 1.0 - w_deform
#                 combined_sim = w_geom * geom_sim + w_deform * deform_sim
#             else:
#                 combined_sim = 1.0 * geom_sim
                
#             cost = 1.0 - combined_sim

#             # if use_oamm:
#             #     occ_score = getattr(track, 'occlusion_score', 0.0)
#             #     if occ_score > 0.15 and geom_sim > 0.05:
#             #         cost_discount = min(0.4 * occ_score * geom_sim, 0.20)
#             #         cost -= cost_discount
#             # if use_oamm:
#             #     occ_score = getattr(track, 'occlusion_score', 0.0)
#             #     # 当发生严重遮挡，且几何交并比 (geom_sim) 不是很完美时：
#             #     if occ_score > 0.15 and geom_sim < 0.6:
#             #         # 增加惩罚，宁愿让它暂时跟丢，也绝不允许它张冠李戴跟错猪！
#             #         cost += 0.2 * occ_score
#             # if use_oamm:
#             #     occ_score = getattr(track, 'occlusion_score', 0.0)
#             #     if occ_score > 0.1 and geom_sim > 0.05:
#             #         cost_discount = min(0.5 * occ_score * geom_sim, 0.30)
#             #         cost -= cost_discount
#             # if use_oamm:
#             #     occ_score = getattr(track, 'occlusion_score', 0.0)
#             #     if occ_score > 0.1:
#             #         if geom_sim < 0.25:
#             #             # 距离远但发生遮挡？绝不妥协，严厉惩罚，绝对防止张冠李戴！
#             #             cost += 0.5 * occ_score 
#             #         elif geom_sim > 0.45:
#             #             # 距离近但因为遮挡导致 IoU 变小？给予精准微弱补偿，保住轨迹不断！
#             #             cost -= 0.05 * occ_score
#             if use_oamm:
#                 occ_score = getattr(track, 'occlusion_score', 0.0)
#                 if occ_score > 0.1:
#                     if geom_sim > 0.65:
#                         # 场景1：预测框和实际框高度重合 (猪在原地没动，只是被压住了)
#                         # 此时给予强力补偿，坚决保住轨迹不断！
#                         cost -= 0.15 * occ_score
#                     elif geom_sim < 0.35:
#                         # 场景2：预测框跑远了 (卡尔曼滤波发生了“线性漂移”，错认了别的猪)
#                         # 此时必须降维打击，施加极其严厉的惩罚，宁愿断开也绝不允许串线！
#                         cost += 1.0 * occ_score

#             cost_matrix[row, col] = min(max(cost, 0.0), linear_assignment.INFTY_COST)
#             col += 1
#     return cost_matrix

# def iou_cost_fuse_score(tracks, detections, track_indices=None,
#                         detection_indices=None, use_riou=False,
#                         use_deform=False, deform_reg=None, track_templates=None,
#                         prev_gray=None, curr_gray=None, use_signature=False,
#                         use_motion_memory=False, use_oamm=False, use_rifm=False,
#                         rifm_extractor=None, frame_id=None, use_bev=False, fqa_gate=None): 
    
#     if track_indices is None:
#         track_indices = np.arange(len(tracks))
#     if detection_indices is None:
#         detection_indices = np.arange(len(detections))

#     cost_matrix = iou_cost(tracks, detections, track_indices, detection_indices,
#                            use_riou, use_deform, deform_reg, track_templates,
#                            prev_gray, curr_gray, use_signature, use_motion_memory,
#                            use_oamm, use_rifm, rifm_extractor, frame_id, use_bev, fqa_gate) 

#     iou_sim = 1 - cost_matrix
#     det_scores = [np.array(det[1].cpu()) if hasattr(det[1], 'cpu') else np.array(det[1]) for det in detections]
#     if len(det_scores) > 0:
#         det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
#         fuse_sim = iou_sim * det_scores
#         return 1 - fuse_sim
#     else:
#         return cost_matrix

# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from tools.track_test import linear_assignment
import shapely
from shapely.geometry import Polygon
import cv2
from tools.track_test.bev_spatial import BEVSpatial

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

def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None, use_riou=False,
             use_deform=False, deform_reg=None, track_templates=None,
             prev_gray=None, curr_gray=None, use_signature=False,
             use_motion_memory=False, use_oamm=False, use_rifm=False, 
             rifm_extractor=None, frame_id=None, use_bev=False, fqa_gate=None): 
    
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    det_rifm_cache = {} 

    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 60:
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        track = tracks[track_idx]
        track_rbox = track.get_rbox()
        track_rbox = np.array(track_rbox, dtype=np.float32)
        is_occluded = use_oamm and getattr(track, 'occlusion_score', 0.0) > 0.08

        col = 0
        for i in detection_indices:
            det_box = detections[i][0]
            det_box = np.array(det_box, dtype=np.float32)

            if use_riou:
                trk_8 = rbox_to_polygon(track_rbox) if len(track_rbox) == 5 else track_rbox.flatten()
                det_8 = rbox_to_polygon(det_box) if len(det_box) == 5 else det_box.flatten()
                iou = iou_eight(trk_8, det_8)
            else:
                iou = compute_horizontal_iou(track_rbox, det_box)

            if use_bev:
                bev_sim = BEVSpatial.compute_bev_similarity(track_rbox, det_box)
                geom_sim = iou + bev_sim - (iou * bev_sim)
            else:
                geom_sim = iou

            deform_sim = 1.0
            if use_deform and deform_reg and track_templates and prev_gray is not None and curr_gray is not None:
                if 0.4 < geom_sim < 0.6 and not is_occluded:
                    track_id = tracks[track_idx].track_id
                    if track_id in track_templates:
                        template_info = track_templates[track_id]
                        if template_info['age'] <= 5:
                            if fqa_gate:
                                candidate_patch = deform_reg.extract_patch(fqa_gate.fast_downsample(curr_gray), fqa_gate.scale_box(det_box))
                            else:
                                candidate_patch = deform_reg.extract_patch(curr_gray, det_box)
                                
                            if candidate_patch is not None and template_info['patch'] is not None:
                                sim = deform_reg.compute_similarity(template_info['patch'], candidate_patch)
                                deform_sim = sim if sim > 0.8 else 1.0

            rifm_sim = 0.0
            if use_rifm and track.rifm_feature is not None and curr_gray is not None and deform_reg is not None:
                if 0.15 < geom_sim < 0.55:  
                    if i not in det_rifm_cache:
                        if fqa_gate:
                            det_patch = deform_reg.extract_patch(fqa_gate.fast_downsample(curr_gray), fqa_gate.scale_box(det_box))
                        else:
                            det_patch = deform_reg.extract_patch(curr_gray, det_box)
                            
                        if det_patch is not None and rifm_extractor is not None:
                            det_rifm_cache[i] = rifm_extractor.extract_features(det_patch)
                        else:
                            det_rifm_cache[i] = None
                    
                    det_feat = det_rifm_cache[i]
                    if det_feat is not None:
                        sim = rifm_extractor.compute_similarity(track.rifm_feature, det_feat)
                        if sim > 0.75:
                            rifm_sim = sim

            occ_score = getattr(track, 'occlusion_score', 0.0) if use_oamm else 0.0

            if use_deform and use_rifm:
                if rifm_sim > 0:
                    w_vis = max(0.0, 0.5 - occ_score) 
                    w_deform = w_vis * (0.2 / 0.5)
                    w_rifm = w_vis * (0.3 / 0.5)
                    w_geom = 1.0 - w_vis
                    combined_sim = w_geom * geom_sim + w_deform * deform_sim + w_rifm * rifm_sim
                else:
                    w_deform = max(0.0, 0.3 - occ_score)
                    w_geom = 1.0 - w_deform
                    combined_sim = w_geom * geom_sim + w_deform * deform_sim
            elif use_deform:
                decay_factor = np.exp(-1.5 * occ_score)
                w_deform = 0.4 * decay_factor
                w_geom = 1.0 - w_deform
                combined_sim = w_geom * geom_sim + w_deform * deform_sim
            else:
                combined_sim = 1.0 * geom_sim
                
            cost = 1.0 - combined_sim

            # ====================================================================
            # 🌟 路线 B 核心创新：挤压唤醒机制 (Squeeze-and-Awake)
            # ====================================================================
            # if use_oamm:
            #     occ_score = getattr(track, 'occlusion_score', 0.0)
            #     if occ_score > 0.15:
            #         if geom_sim > 0.50:
            #             # 场景1：猪在原地没动被重新发现！给予高额奖励，瞬间唤醒轨迹！
            #             cost -= 0.3 * occ_score
            #         elif geom_sim < 0.25:
            #             # 场景2：框跑远了！这绝对是卡尔曼串线，直接判死刑，禁止匹配！
            #             cost += 2.0 * occ_score
            # ====================================================================
            if use_oamm:
                occ_score = getattr(track, 'occlusion_score', 0.0)
                if occ_score > 0.15 and geom_sim > 0.20:
                    cost_discount = 0.5 * occ_score * geom_sim
                    cost -= cost_discount

            cost_matrix[row, col] = min(max(cost, 0.0), linear_assignment.INFTY_COST)
            col += 1
    return cost_matrix

def iou_cost_fuse_score(tracks, detections, track_indices=None,
                        detection_indices=None, use_riou=False,
                        use_deform=False, deform_reg=None, track_templates=None,
                        prev_gray=None, curr_gray=None, use_signature=False,
                        use_motion_memory=False, use_oamm=False, use_rifm=False,
                        rifm_extractor=None, frame_id=None, use_bev=False, fqa_gate=None): 
    
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = iou_cost(tracks, detections, track_indices, detection_indices,
                           use_riou, use_deform, deform_reg, track_templates,
                           prev_gray, curr_gray, use_signature, use_motion_memory,
                           use_oamm, use_rifm, rifm_extractor, frame_id, use_bev, fqa_gate) 

    iou_sim = 1 - cost_matrix
    det_scores = [np.array(det[1].cpu()) if hasattr(det[1], 'cpu') else np.array(det[1]) for det in detections]
    if len(det_scores) > 0:
        det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
        fuse_sim = iou_sim * det_scores
        return 1 - fuse_sim
    else:
        return cost_matrix