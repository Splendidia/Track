"""Run DBPig-ORCNN detector on image sequence and save detections to txt files."""
import os
import cv2
import glob
import torch
import numpy as np
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector
import mmrotate  # noqa: F401
from tqdm import tqdm
from mmengine import Config

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--image-dir', required=True,
                        help='Directory containing frame images (jpg)')
    parser.add_argument('--config', required=True,
                        help='Config file path')
    parser.add_argument('--checkpoint', required=True,
                        help='Checkpoint file path')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--score-thr', type=float, default=0.3,
                        help='Score threshold')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory for detection txt files')
    args = parser.parse_args()
    return args

def rbox_to_polygon(rbox):
    """Convert [x, y, w, h, angle] (radians) to 8-point coordinates."""
    x, y, w, h, angle = rbox
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

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 注册模块
    from mmrotate.utils import register_all_modules
    from mmdet.utils import register_all_modules as register_all_modules_mmdet
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)

    # 加载模型
    print("Loading model...")
    model = init_detector(args.config, args.checkpoint, device=args.device)
    print("Model loaded.")

    # 获取类别名称（优先从配置文件读取）
    try:
        cfg = Config.fromfile(args.config)
        classes = cfg.get('classes', ['sow', 'piglet', 'placenta'])
    except:
        classes = ['sow', 'piglet', 'placenta']
    print(f"Classes: {classes}")

    # 获取所有图片文件，按文件名排序
    image_files = sorted(glob.glob(os.path.join(args.image_dir, '*.jpg')))
    print(f"Found {len(image_files)} images.")

    for img_path in tqdm(image_files, desc="Processing"):
        # 提取帧号（假设文件名如 frame_00001.jpg）
        basename = os.path.basename(img_path)
        frame_id = int(basename.split('_')[1].split('.')[0])
        out_txt = os.path.join(args.out_dir, f'frame_{frame_id:06d}.txt')

        # 读取图像并推理
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Warning: cannot read {img_path}, skip.")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = inference_detector(model, img_rgb)

        # 提取检测结果
        pred_instances = result.pred_instances
        if args.score_thr > 0:
            keep = pred_instances.scores > args.score_thr
            pred_instances = pred_instances[keep]

        bboxes = pred_instances.bboxes.cpu().numpy()  # (N, 5) or (N, 8)
        scores = pred_instances.scores.cpu().numpy()
        labels = pred_instances.labels.cpu().numpy()

        # 写入txt文件，格式：8点坐标 + class_name + score
        with open(out_txt, 'w') as f:
            for bbox, score, label in zip(bboxes, scores, labels):
                if len(bbox) == 5:
                    pts = rbox_to_polygon(bbox)
                else:
                    pts = bbox.flatten()
                class_name = classes[int(label)]
                line = ' '.join([f'{p:.2f}' for p in pts]) + f' {class_name} {score:.4f}\n'
                f.write(line)

    print(f"Detections saved to {args.out_dir}")

if __name__ == '__main__':
    main()