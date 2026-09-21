import os
import json
import glob
import numpy as np
import cv2
import motmetrics as mm
import sys
import argparse
import time
from shapely.geometry import Polygon

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from tools.track_test.byte_tracker import Byte_tracker
from tools.track_test.trajectory_postprocess import TrajectoryProcessor 

def load_detections(video_folder):
    # det_dir = os.path.join(video_folder, 'detections_yolo26') 
    det_dir = os.path.join(video_folder, 'detections')
    det_files = sorted(glob.glob(os.path.join(det_dir, 'frame_*.txt')))
    frames = {}
    for det_file in det_files:
        frame_id = int(os.path.basename(det_file).split('_')[1].split('.')[0])
        boxes, scores, classes = [], [], []
        with open(det_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10: continue
                coords = list(map(float, parts[:8]))
                class_name = parts[8]
                score = float(parts[9])
                cls_id = 0 if class_name == 'sow' else (1 if class_name == 'piglet' else (2 if class_name == 'placenta' else -1))
                boxes.append(coords); scores.append(score); classes.append(cls_id)
        frames[frame_id] = {'boxes': boxes, 'scores': scores, 'classes': classes}
    return frames

def load_gt(video_folder):
    frames = {}
    json_files = sorted(glob.glob(os.path.join(video_folder, '*.json')))
    for jf in json_files:
        with open(jf) as f: data = json.load(f)
        frame_id = int(data['imagePath'].split('_')[1].split('.')[0])
        ids, boxes, classes = [], [], []
        for shape in data['shapes']:
            label = shape['label']
            ids.append(shape.get('group_id', 0))
            boxes.append(np.array(shape['points']).flatten().tolist())
            classes.append(0 if label == 'sow' else (1 if label == 'piglet' else (2 if label == 'placenta' else -1)))
        frames[frame_id] = {'ids': ids, 'boxes': boxes, 'classes': classes}
    return frames

# def run_tracker(video_folder, tracker, use_gt=False):
#     det_frames = load_gt(video_folder) if use_gt else load_detections(video_folder)
#     if use_gt:
#         for fid in det_frames: det_frames[fid]['scores'] = [1.0] * len(det_frames[fid]['boxes'])
    
#     total_track_time, frame_count = 0.0, 0
#     flat_results = [] 
    
#     for frame_id in sorted(det_frames.keys()):
#         img_path = os.path.join(video_folder, f'frame_{frame_id:05d}.jpg')
#         img = cv2.imread(img_path) if os.path.exists(img_path) else None
        
#         det_info = det_frames.get(frame_id, {'boxes':[], 'scores':[], 'classes':[]})
        
#         start_time = time.time()
#         tracking_results = tracker.update(det_info['boxes'], det_info['scores'], det_info['classes'], img, frame_id)
#         total_track_time += (time.time() - start_time)
#         frame_count += 1
        
#         for rbox, tid, cid, _ in tracking_results:
#             flat_results.append([frame_id, tid] + list(rbox))

#     stitched_results = TrajectoryProcessor.sbs_stitching(
#         flat_results, max_stitch_gap=60, max_interp_gap=30
#     )
#     pred = TrajectoryProcessor.apply_gtf_and_tbs(
#         stitched_results, min_life=3, alpha=0.4
#     )

#     fps = frame_count / total_track_time if total_track_time > 0 else 0
#     print(f"  [Time Log] {frame_count} frames | Time: {total_track_time:.2f}s | Speed: {fps:.1f} FPS")
#     return pred

def run_tracker(video_folder, tracker, use_gt=False, use_postproc=False):
    det_frames = load_gt(video_folder) if use_gt else load_detections(video_folder)
    if use_gt:
        for fid in det_frames: det_frames[fid]['scores'] = [1.0] * len(det_frames[fid]['boxes'])
    
    total_track_time, frame_count = 0.0, 0
    flat_results = [] 
    
    for frame_id in sorted(det_frames.keys()):
        img_path = os.path.join(video_folder, f'frame_{frame_id:05d}.jpg')
        img = cv2.imread(img_path) if os.path.exists(img_path) else None
        
        det_info = det_frames.get(frame_id, {'boxes':[], 'scores':[], 'classes':[]})
        
        start_time = time.time()
        tracking_results = tracker.update(det_info['boxes'], det_info['scores'], det_info['classes'], img, frame_id)
        total_track_time += (time.time() - start_time)
        frame_count += 1
        
        for rbox, tid, cid, _ in tracking_results:
            flat_results.append([frame_id, tid] + list(rbox))

    # 🌟 核心：只有开启外挂时，才执行你的 SOTA 后处理
    if use_postproc:
        stitched_results = TrajectoryProcessor.sbs_stitching(flat_results, max_stitch_gap=60, max_interp_gap=30)
        pred = TrajectoryProcessor.apply_gtf_and_tbs(stitched_results, min_life=3, alpha=0.4)
    else:
        # 裸跑模式：原汁原味的在线跟踪结果，展示算法真实功底！
        pred = {}
        for res in flat_results:
            fid, tid = int(res[0]), int(res[1])
            box = res[2:7]
            if fid not in pred:
                pred[fid] = {'ids': [], 'boxes': []}
            pred[fid]['ids'].append(tid)
            pred[fid]['boxes'].append(box)

    fps = frame_count / total_track_time if total_track_time > 0 else 0
    print(f"  [Time Log] {frame_count} frames | Time: {total_track_time:.2f}s | Speed: {fps:.1f} FPS")
    return pred

def rbox_to_polygon(rbox):
    x, y, w, h, angle = rbox[:5]
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    half_w, half_h = w / 2, h / 2
    corners = np.array([[-half_w, -half_h], [half_w, -half_h], [half_w, half_h], [-half_w, half_h]])
    rotated = np.dot(corners, np.array([[cos_a, -sin_a], [sin_a, cos_a]]).T)
    rotated[:, 0] += x; rotated[:, 1] += y
    return rotated.flatten()

def compute_rotated_iou(pts1, pts2):
    try:
        poly1 = Polygon(np.array(pts1).reshape(4, 2)).convex_hull
        poly2 = Polygon(np.array(pts2).reshape(4, 2)).convex_hull
        if not poly1.intersects(poly2): return 0.0
        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0

def compute_horizontal_iou(pts1, pts2):
    rect1 = cv2.boundingRect(np.array(pts1).reshape(4, 2).astype(np.int32))
    rect2 = cv2.boundingRect(np.array(pts2).reshape(4, 2).astype(np.int32))
    x1, y1 = max(rect1[0], rect2[0]), max(rect1[1], rect2[1])
    x2, y2 = min(rect1[0]+rect1[2], rect2[0]+rect2[2]), min(rect1[1]+rect1[3], rect2[1]+rect2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    union = (rect1[2] * rect1[3]) + (rect2[2] * rect2[3]) - inter
    return inter / union if union > 0 else 0

def compute_iou_matrix(gt_boxes, pred_boxes, use_riou=True):
    dist = np.full((len(gt_boxes), len(pred_boxes)), np.nan)
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            pred_8 = rbox_to_polygon(pred) if len(pred) == 5 else np.array(pred).flatten()
            iou = compute_rotated_iou(gt, pred_8) if use_riou else compute_horizontal_iou(gt, pred_8)
            if iou > 0.05: 
                dist[i, j] = 1.0 - iou
    return dist

def evaluate_video(gt_frames, pred_frames, use_riou=False):
    acc = mm.MOTAccumulator(auto_id=True)
    for fid in sorted(set(gt_frames.keys()) | set(pred_frames.keys())):
        gt, pr = gt_frames.get(fid, {'ids': [], 'boxes': []}), pred_frames.get(fid, {'ids': [], 'boxes': []})
        if len(gt['ids']) == 0 and len(pr['ids']) == 0: continue
        acc.update(gt['ids'], pr['ids'], compute_iou_matrix(gt['boxes'], pr['boxes'], use_riou=use_riou))
    return acc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default=r'E:\pigtrackvideo')
    parser.add_argument('--video_ids', type=int, nargs='+', required=True)
    parser.add_argument('--iou_threshold', type=float, default=0.85)
    parser.add_argument('--max_age', type=int, default=30)
    parser.add_argument('--n_init', type=int, default=3)
    parser.add_argument('--use_riou', action='store_true')
    parser.add_argument('--use_deform', action='store_true')
    parser.add_argument('--use_oamm', action='store_true')
    parser.add_argument('--use_sgmp', action='store_true')
    parser.add_argument('--use_fac', action='store_true')
    parser.add_argument('--use_rifm', action='store_true')
    parser.add_argument('--use_bev', action='store_true') 
    parser.add_argument('--use_postproc', action='store_true')
    parser.add_argument('--exp_name', type=str, default='default_exp')
    return parser.parse_args()

def main():
    args = parse_args()
    video_folders = [os.path.join(args.base_dir, str(i)) for i in args.video_ids]

    print(f"\n=== Experiment: {args.exp_name} ===")
    metrics_list = []
    
    for vf in video_folders:
        if not os.path.exists(vf): continue
        print(f"Processing {vf}...")
        
        tracker = Byte_tracker(
            max_iou_distance=args.iou_threshold, max_age=args.max_age, n_init=args.n_init,
            use_riou=args.use_riou, use_deform=args.use_deform, 
            use_oamm=args.use_oamm, # 传入 use_oamm
            use_sgmp=args.use_sgmp, # 传入 use_sgmp
            use_fac=args.use_fac, use_rifm=args.use_rifm, n_sig_components=5,
            use_bev=args.use_bev 
        )
        
        gt = load_gt(vf)
        # pred = run_tracker(vf, tracker, use_gt=False) 
        pred = run_tracker(vf, tracker, use_gt=True, use_postproc=args.use_postproc)
        acc = evaluate_video(gt, pred, use_riou=args.use_riou)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'motp'], name='acc')
        metrics_list.append(summary)

    if len(metrics_list) == 0: return

    print(f"\n--- Final Results for {args.exp_name} ---")
    print(f"MOTA: {np.mean([m['mota'] for m in metrics_list]):.3f}")
    print(f"IDF1: {np.mean([m['idf1'] for m in metrics_list]):.3f}")
    print(f"IDSW: {np.mean([m['num_switches'] for m in metrics_list]):.1f}")
    print(f"MOTP: {np.mean([m['motp'] for m in metrics_list]):.3f}")

if __name__ == '__main__':
    main()