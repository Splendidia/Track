import os
import json
import glob
import numpy as np
import cv2
import motmetrics as mm
import sys
import argparse
sys.path.insert(0, r'E:\project\track')  # 请根据实际路径修改

from tools.track_test.byte_tracker import Byte_tracker
from shapely.geometry import Polygon

# ========== 辅助函数（与之前相同） ==========
def load_detections(video_folder):
    det_dir = os.path.join(video_folder, 'detections')
    det_files = sorted(glob.glob(os.path.join(det_dir, 'frame_*.txt')))
    frames = {}
    for det_file in det_files:
        frame_id = int(os.path.basename(det_file).split('_')[1].split('.')[0])
        boxes = []
        scores = []
        classes = []
        with open(det_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                coords = list(map(float, parts[:8]))
                class_name = parts[8]
                score = float(parts[9])
                if class_name == 'sow':
                    cls_id = 0
                elif class_name == 'piglet':
                    cls_id = 1
                elif class_name == 'placenta':
                    cls_id = 2
                else:
                    cls_id = -1
                boxes.append(coords)
                scores.append(score)
                classes.append(cls_id)
        frames[frame_id] = {'boxes': boxes, 'scores': scores, 'classes': classes}
    return frames

def load_gt(video_folder):
    frames = {}
    json_files = sorted(glob.glob(os.path.join(video_folder, '*.json')))
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        frame_name = data['imagePath']
        frame_id = int(frame_name.split('_')[1].split('.')[0])
        ids, boxes, classes = [], [], []
        for shape in data['shapes']:
            label = shape['label']
            points = np.array(shape['points']).flatten().tolist()
            ids.append(shape.get('group_id', 0))
            boxes.append(points)
            if label == 'sow':
                cls_id = 0
            elif label == 'piglet':
                cls_id = 1
            elif label == 'placenta':
                cls_id = 2
            else:
                cls_id = -1
            classes.append(cls_id)
        frames[frame_id] = {'ids': ids, 'boxes': boxes, 'classes': classes}
    return frames

def run_tracker(video_folder, tracker, use_gt=False):
    if use_gt:
        det_frames = load_gt(video_folder)
        for fid in det_frames:
            det_frames[fid]['scores'] = [1.0] * len(det_frames[fid]['boxes'])
    else:
        det_frames = load_detections(video_folder)

    pred = {}
    frame_ids = sorted(det_frames.keys())
    for frame_id in frame_ids:
        img_path = os.path.join(video_folder, f'frame_{frame_id:05d}.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
        else:
            img = None

        det_info = det_frames[frame_id]
        detections = det_info['boxes']
        confidences = det_info['scores']
        classes = det_info['classes']

        tracking_results = tracker.update(detections, confidences, classes, img)
        pred_ids = []
        pred_boxes = []
        for res in tracking_results:
            rbox, tid, cid, _ = res
            pred_ids.append(tid)
            pred_boxes.append(rbox)
        pred[frame_id] = {'ids': pred_ids, 'boxes': pred_boxes}
    return pred

def rbox_to_polygon(rbox):
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

def compute_rotated_iou(pts1, pts2):
    poly1 = Polygon(np.array(pts1).reshape(4, 2))
    poly2 = Polygon(np.array(pts2).reshape(4, 2))
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.area + poly2.area - inter
    return inter / union if union > 0 else 0.0

def compute_horizontal_iou(pts1, pts2):
    rect1 = cv2.boundingRect(np.array(pts1).reshape(4, 2).astype(np.int32))
    rect2 = cv2.boundingRect(np.array(pts2).reshape(4, 2).astype(np.int32))
    x1 = max(rect1[0], rect2[0])
    y1 = max(rect1[1], rect2[1])
    x2 = min(rect1[0]+rect1[2], rect2[0]+rect2[2])
    y2 = min(rect1[1]+rect1[3], rect2[1]+rect2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = rect1[2] * rect1[3]
    area2 = rect2[2] * rect2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def compute_iou_matrix(gt_boxes, pred_boxes, use_riou=True):
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    dist = np.full((n_gt, n_pred), np.nan)
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            pred_8 = rbox_to_polygon(pred) if len(pred) == 5 else np.array(pred).flatten()
            if use_riou:
                iou = compute_rotated_iou(gt, pred_8)
            else:
                iou = compute_horizontal_iou(gt, pred_8)
            dist[i, j] = 1 - iou
    return dist

def evaluate_video(gt_frames, pred_frames):
    acc = mm.MOTAccumulator(auto_id=True)
    all_frame_ids = sorted(set(gt_frames.keys()) | set(pred_frames.keys()))
    for fid in all_frame_ids:
        gt = gt_frames.get(fid, {'ids': [], 'boxes': []})
        pr = pred_frames.get(fid, {'ids': [], 'boxes': []})
        if len(gt['ids']) == 0 and len(pr['ids']) == 0:
            continue
        dist = compute_iou_matrix(gt['boxes'], pr['boxes'], use_riou=True)
        acc.update(gt['ids'], pr['ids'], dist)
    return acc

def parse_args():
    parser = argparse.ArgumentParser(description='Run ablation experiments for ByteTrack with various modules.')
    parser.add_argument('--base_dir', type=str, default='/home/dy/NewDisk/pigtrackvideo',
                        help='Base directory containing video subfolders (1,2,3,...)')
    parser.add_argument('--video_ids', type=int, nargs='+', default=None,
                        help='List of video IDs to process (e.g., 1 2 3). If not specified, all videos 1-10 are used.')
    parser.add_argument('--iou_threshold', type=float, default=0.95,
                        help='Matching IoU threshold (max_iou_distance)')
    parser.add_argument('--max_age', type=int, default=70,
                        help='Maximum age for lost tracks')
    parser.add_argument('--n_init', type=int, default=3,
                        help='Number of frames to confirm a track')
    parser.add_argument('--use_riou', action='store_true',
                        help='Use rotated IoU')
    parser.add_argument('--use_10d_kf', action='store_true',
                        help='Use 10D Kalman filter (includes angle)')
    parser.add_argument('--use_imm', action='store_true',
                        help='Use IMM filter (requires --use_10d_kf)')
    parser.add_argument('--use_deform', action='store_true',
                        help='Use deformable registration')
    parser.add_argument('--use_signature', action='store_true',
                        help='Use motion signature')
    parser.add_argument('--n_sig_components', type=int, default=5,
                        help='Number of signature components (if --use_signature)')
    parser.add_argument('--exp_name', type=str, default='experiment',
                        help='Name of this experiment (for printing)')
    return parser.parse_args()

def main():
    args = parse_args()

    # 确定要处理的视频文件夹
    if args.video_ids is not None:
        video_folders = [os.path.join(args.base_dir, str(i)) for i in args.video_ids]
    else:
        video_folders = [os.path.join(args.base_dir, str(i)) for i in range(1, 11)]

    # 构造跟踪器
    tracker = Byte_tracker(
        max_iou_distance=args.iou_threshold,
        max_age=args.max_age,
        n_init=args.n_init,
        use_riou=args.use_riou,
        use_10d_kf=args.use_10d_kf,
        use_imm=args.use_imm,
        use_deform=args.use_deform,
        use_signature=args.use_signature,
        n_sig_components=args.n_sig_components
    )

    print(f"\n=== Running experiment: {args.exp_name} ===")
    print(f"Settings: use_riou={args.use_riou}, use_10d_kf={args.use_10d_kf}, use_imm={args.use_imm}, use_deform={args.use_deform}, use_signature={args.use_signature}")
    print(f"iou_threshold={args.iou_threshold}, max_age={args.max_age}, n_init={args.n_init}")

    metrics_list = []
    for vf in video_folders:
        if not os.path.exists(vf):
            print(f"Warning: {vf} does not exist, skipping.")
            continue
        print(f"Processing {vf}")
        gt = load_gt(vf)
        pred = run_tracker(vf, tracker, use_gt=False)
        acc = evaluate_video(gt, pred)
        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches', 'motp'], name='acc')
        metrics_list.append(summary)

    if len(metrics_list) == 0:
        print("No valid videos processed.")
        return

    avg_mota = np.mean([m['mota'] for m in metrics_list])
    avg_idf1 = np.mean([m['idf1'] for m in metrics_list])
    avg_idsw = np.mean([m['num_switches'] for m in metrics_list])
    avg_motp = np.mean([m['motp'] for m in metrics_list])

    print(f"\n=== Results for {args.exp_name} ===")
    print(f"MOTA: {avg_mota:.3f}")
    print(f"IDF1: {avg_idf1:.3f}")
    print(f"IDSW: {avg_idsw:.1f}")
    print(f"MOTP: {avg_motp:.3f}")

if __name__ == '__main__':
    main()