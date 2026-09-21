"""Inference on a video file with real-time display and BYTE tracking."""
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector
import mmrotate  # noqa: F401
import os
import cv2
import glob
from pathlib import Path
import time
import numpy as np
import torch

# 导入跟踪模块
import sys
sys.path.append('.')  # 确保可以导入本地模块
from tools.byte_tracker import Byte_tracker

ROOT = os.getcwd()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--video-path', 
                       required=True,
                       help='Video file path or camera index (e.g., 0 for webcam)')
    parser.add_argument('--config', 
                       default='/home/dy/下载/OrientedFormer/projects/OrientedFormer/orientedformer_config.py', 
                       help='Config file path')
    parser.add_argument('--checkpoint', 
                       default='/home/dy/下载/OrientedFormer/work_dirs/pig_le90_r50_q300_layer2_head64_point32_1x_pig/best_dota_AP50_epoch_24.pth', 
                       help='Checkpoint file path')
    parser.add_argument('--device', 
                       default='cuda:0', 
                       help='Device used for inference (cuda:0 or cpu)')
    parser.add_argument('--palette', 
                       default='dota',
                       choices=['dota', 'sar', 'hrsc', 'random'], 
                       help='Color palette for visualization')
    parser.add_argument('--score-thr', 
                       type=float, 
                       default=0.3, 
                       help='Bounding box score threshold')
    parser.add_argument('--track-high-thresh', 
                       type=float, 
                       default=0.5, 
                       help='High confidence threshold for tracking')
    parser.add_argument('--track-low-thresh', 
                       type=float, 
                       default=0.1, 
                       help='Low confidence threshold for tracking')
    parser.add_argument('--match-thresh', 
                       type=float, 
                       default=0.7, 
                       help='Matching threshold for tracking')
    parser.add_argument('--n-init', 
                       type=int, 
                       default=3,
                       help='Number of frames to confirm a track')
    parser.add_argument('--max-age', 
                       type=int, 
                       default=70,
                       help='Maximum age of lost tracks')
    parser.add_argument('--out-dir', 
                       type=str, 
                       default='./video_results', 
                       help='Directory to save output video and results')
    parser.add_argument('--save-video', 
                       action='store_true',
                       help='Save output video with detections')
    parser.add_argument('--save-txt', 
                       action='store_true',
                       help='Save detection results to txt files (per frame)')
    parser.add_argument('--save-tracks', 
                       action='store_true',
                       help='Save tracking results to txt files')
    parser.add_argument('--no-display', 
                       action='store_true',
                       help='Do not display video during inference (for headless mode)')
    parser.add_argument('--frame-skip', 
                       type=int, 
                       default=0,
                       help='Skip every N frames to speed up processing')
    parser.add_argument('--fps', 
                       type=float, 
                       default=None,
                       help='Output video FPS (default: same as input video)')
    parser.add_argument('--show-fps', 
                       action='store_true',
                       help='Show FPS counter on video')
    parser.add_argument('--show-ids', 
                       action='store_true',
                       default=True,
                       help='Show track IDs on video')
    args = parser.parse_args()
    return args

def rbox2polygon(x, y, w, h, angle):
    """Convert rotated rectangle (x, y, w, h, angle) to polygon points."""
    # angle in radians
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Half dimensions
    half_w = w / 2
    half_h = h / 2
    
    # Four corners relative to center
    corners = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ])
    
    # Rotation matrix
    rotation_matrix = np.array([[cos_a, -sin_a],
                                 [sin_a, cos_a]])
    
    # Rotate corners
    rotated_corners = np.dot(corners, rotation_matrix.T)
    
    # Translate to (x, y)
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y
    
    return rotated_corners

def draw_rotated_bbox(img, bbox, color, thickness=2):
    """Draw rotated bounding box on image."""
    # bbox format: [x_center, y_center, width, height, angle] (angle in radians)
    if len(bbox) == 5:  # Rotated rectangle format
        x, y, w, h, angle = bbox
        points = rbox2polygon(x, y, w, h, angle)
        points = points.astype(np.int32)
        
        # Draw the bounding box edges
        for i in range(4):
            pt1 = tuple(points[i])
            pt2 = tuple(points[(i + 1) % 4])
            cv2.line(img, pt1, pt2, color, thickness)
        
        # Draw corner points
        for point in points:
            cv2.circle(img, tuple(point), 4, color, -1)
            
        return points, (x, y, w, h)
    elif len(bbox) == 8:  # Quadrilateral format
        points = bbox.reshape(4, 2).astype(np.int32)
        
        # Draw the bounding box edges
        for i in range(4):
            pt1 = tuple(points[i])
            pt2 = tuple(points[(i + 1) % 4])
            cv2.line(img, pt1, pt2, color, thickness)
        
        # Draw corner points
        for point in points:
            cv2.circle(img, tuple(point), 4, color, -1)
        
        # 计算中心点用于显示ID
        x_center = int(np.mean(points[:, 0]))
        y_center = int(np.mean(points[:, 1]))
        w = np.sqrt(np.sum((points[0] - points[1]) ** 2))
        h = np.sqrt(np.sum((points[1] - points[2]) ** 2))
        
        return points, (x_center, y_center, w, h)
    else:
        print(f"Warning: Unknown bbox format with length {len(bbox)}")
        return None, (0, 0, 0, 0)

def main(args):
    print("=" * 60)
    print("🎥 Pig Detection Inference on Video with BYTE Tracking")
    print("=" * 60)
    
    # ============ 创建输出文件夹 ============
    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_txt:
        txt_dir = os.path.join(args.out_dir, 'detections')
        os.makedirs(txt_dir, exist_ok=True)
    
    if args.save_tracks:
        tracks_dir = os.path.join(args.out_dir, 'tracks')
        os.makedirs(tracks_dir, exist_ok=True)
    
    # ============ 注册所有模块 ============
    from mmrotate.utils import register_all_modules
    from mmdet.utils import register_all_modules as register_all_modules_mmdet
    
    register_all_modules_mmdet(init_default_scope=False)
    register_all_modules(init_default_scope=False)
    
    print(f"📁 视频路径: {args.video_path}")
    print(f"💾 输出文件夹: {args.out_dir}")
    print(f"🎯 跟踪高阈值: {args.track_high_thresh}")
    print(f"🎯 跟踪低阈值: {args.track_low_thresh}")
    print(f"🔗 匹配阈值: {args.match_thresh}")
    print(f"🔄 确认帧数: {args.n_init}")
    print(f"⏰ 最大丢失帧数: {args.max_age}")
    
    # ============ 初始化跟踪器 ============
    tracker = Byte_tracker(
        max_iou_distance=args.match_thresh,
        max_age=args.max_age,
        n_init=args.n_init
    )
    
    print(f"✅ BYTE跟踪器初始化完成")
    
    # ============ 加载模型 ============
    print("\n🔄 正在加载模型...")
    
    try:
        # 构建模型
        model = init_detector(
            args.config, 
            args.checkpoint, 
            device=args.device
        )
        print(f"✅ 模型加载成功，使用设备: {args.device}")
        
        # 获取类别信息
        if hasattr(model, 'dataset_meta') and model.dataset_meta:
            classes = model.dataset_meta.get('classes', ['unknown'])
        else:
            # 从配置文件中读取classes
            from mmengine import Config
            cfg = Config.fromfile(args.config)
            classes = cfg.get('classes', ['sow', 'piglet', 'placenta'])
        
        print(f"📋 检测类别: {classes}")
        
        # 创建颜色映射
        colors = {}
        for i, class_name in enumerate(classes):
            if args.palette == 'random':
                colors[i] = tuple(np.random.randint(0, 255, 3).tolist())
            else:
                # 使用固定颜色
                color_palette = {
                    0: (0, 255, 0),    # 绿色 - sow
                    1: (255, 0, 0),    # 红色 - piglet
                    2: (0, 0, 255),    # 蓝色 - placenta
                }
                colors[i] = color_palette.get(i, (255, 255, 0))
            
            # 确保颜色是整数元组
            colors[i] = tuple(int(c) for c in colors[i])
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ============ 打开视频文件 ============
    try:
        # 检查是否是摄像头
        if str(args.video_path).isdigit():
            video_path = int(args.video_path)
        else:
            video_path = args.video_path
            if not os.path.exists(video_path):
                print(f"❌ 错误: 视频文件不存在: {video_path}")
                return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 错误: 无法打开视频: {video_path}")
            return
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📊 视频信息:")
        print(f"  帧率: {fps:.2f} FPS")
        print(f"  分辨率: {width}x{height}")
        print(f"  总帧数: {total_frames if total_frames > 0 else 'Unknown'}")
        
        # 设置输出视频参数
        out_video = None
        if args.save_video:
            output_fps = args.fps if args.fps else fps
            if output_fps <= 0:
                output_fps = 30  # 默认值
            
            video_name = Path(str(args.video_path)).stem if not str(args.video_path).isdigit() else 'camera'
            output_path = os.path.join(args.out_dir, f'{video_name}_tracked.mp4')
            
            # 使用合适的编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
            print(f"  输出视频: {output_path} ({output_fps} FPS)")
        
    except Exception as e:
        print(f"❌ 视频打开失败: {e}")
        return
    
    # ============ 初始化统计信息 ============
    print("\n🚀 开始视频推理与跟踪...")
    print("按 'q' 退出，按 'p' 暂停，按 's' 保存当前帧")
    
    frame_count = 0
    processed_frames = 0
    total_detections = 0
    total_tracks = 0
    total_inference_time = 0
    
    # FPS计算
    fps_start_time = time.time()
    fps_frame_count = 0
    display_fps = 0
    
    # 暂停状态
    paused = False
    
    # 跟踪结果保存
    all_tracks = []
    
    # ============ 主循环 ============
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n✅ 视频处理完成!")
                    break
                
                frame_count += 1
                
                # 帧跳过
                if args.frame_skip > 0 and frame_count % (args.frame_skip + 1) != 0:
                    continue
                
                processed_frames += 1
                
                # 转换颜色空间 (BGR -> RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 推理
                inference_start = time.time()
                result = inference_detector(model, frame_rgb)
                inference_time = time.time() - inference_start
                total_inference_time += inference_time
                
                # 提取预测实例
                pred_instances = result.pred_instances
                
                # 根据置信度阈值筛选
                if args.score_thr > 0:
                    keep_idxs = pred_instances.scores > args.score_thr
                    filtered_instances = pred_instances[keep_idxs]
                else:
                    filtered_instances = pred_instances
                
                num_detections = len(filtered_instances)
                total_detections += num_detections
                
                # 准备检测结果用于跟踪
                rboxes_for_tracking = []
                confidences_for_tracking = []
                classes_for_tracking = []
                
                if num_detections > 0:
                    bboxes = filtered_instances.bboxes.cpu().numpy()
                    scores = filtered_instances.scores.cpu().numpy()
                    labels = filtered_instances.labels.cpu().numpy()
                    
                    for bbox, score, label in zip(bboxes, scores, labels):
                        # 检查bbox格式
                        if len(bbox) not in [5, 8]:
                            continue
                        
                        rboxes_for_tracking.append(bbox)
                        confidences_for_tracking.append(float(score))
                        classes_for_tracking.append(int(label))
                
                # 更新跟踪器
                if len(rboxes_for_tracking) > 0:
                    tracking_results = tracker.update(
                        rboxes_for_tracking, 
                        confidences_for_tracking, 
                        classes_for_tracking,
                        frame
                    )
                    
                    num_tracks = len(tracking_results) if tracking_results is not None else 0
                    total_tracks += num_tracks
                    
                    # 绘制跟踪结果
                    if tracking_results is not None and len(tracking_results) > 0:
                        for result in tracking_results:
                            if len(result) == 4:
                                track_bbox, track_id, class_id, confidence = result
                                
                                # 获取颜色
                                color = colors.get(class_id, (0, 255, 255))
                                
                                # 绘制边界框
                                points, (x_center, y_center, w, h) = draw_rotated_bbox(frame, track_bbox, color, 2)
                                
                                # 绘制跟踪ID和类别信息
                                if args.show_ids:
                                    class_name = classes[class_id] if class_id < len(classes) else f'class_{class_id}'
                                    
                                    # ID文本
                                    id_text = f"ID:{track_id}"
                                    id_text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                                    
                                    # 类别和置信度文本
                                    info_text = f"{class_name}:{confidence:.2f}"
                                    info_text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                                    
                                    # 计算文本位置
                                    text_x = max(10, x_center - id_text_size[0]//2)
                                    text_y = max(30, y_center - 30)
                                    
                                    # 绘制ID背景
                                    pt1 = (int(text_x - 5), int(text_y - id_text_size[1] - 5))
                                    pt2 = (int(text_x + id_text_size[0] + 5), int(text_y + 5))
                                    cv2.rectangle(frame, pt1, pt2, color, -1)
                                    
                                    # 绘制ID文本
                                    cv2.putText(frame, id_text,
                                              (int(text_x), int(text_y)),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                    
                                    # 绘制信息文本背景
                                    pt3 = (int(text_x - 5), int(text_y + 5))
                                    pt4 = (int(text_x + info_text_size[0] + 5), int(text_y + info_text_size[1] + 15))
                                    cv2.rectangle(frame, pt3, pt4, color, -1)
                                    
                                    # 绘制信息文本
                                    cv2.putText(frame, info_text,
                                              (int(text_x), int(text_y + info_text_size[1] + 10)),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                # 保存跟踪结果
                                if args.save_tracks:
                                    all_tracks.append({
                                        'frame': frame_count,
                                        'track_id': int(track_id),
                                        'class_id': int(class_id),
                                        'class_name': classes[int(class_id)] if int(class_id) < len(classes) else f'class_{int(class_id)}',
                                        'confidence': float(confidence),
                                        'bbox': track_bbox.tolist() if hasattr(track_bbox, 'tolist') else list(track_bbox)
                                    })
                
                # 保存检测结果到txt
                if args.save_txt and num_detections > 0:
                    txt_path = os.path.join(txt_dir, f'frame_{frame_count:06d}.txt')
                    with open(txt_path, 'w') as f:
                        for bbox, score, label in zip(bboxes, scores, labels):
                            if len(bbox) == 5:  # 旋转矩形格式
                                bbox_str = ' '.join([f'{coord:.2f}' for coord in bbox])
                            else:  # 四边形格式
                                bbox_str = ' '.join([f'{coord:.2f}' for coord in bbox])
                            class_name = classes[int(label)] if int(label) < len(classes) else f'class_{int(label)}'
                            f.write(f'{bbox_str} {class_name} {score:.4f}\n')
                
                # 在帧上添加统计信息
                stats_text = f'Frame: {frame_count} | Tracks: {num_tracks} | FPS: {display_fps:.1f}'
                cv2.putText(frame, stats_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                if args.show_fps:
                    cv2.putText(frame, f'Inference: {inference_time*1000:.1f}ms', (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 保存到输出视频
                if out_video:
                    out_video.write(frame)
                
                # FPS计算
                fps_frame_count += 1
                if fps_frame_count >= 30:  # 每30帧更新一次FPS
                    elapsed = time.time() - fps_start_time
                    display_fps = fps_frame_count / elapsed
                    fps_start_time = time.time()
                    fps_frame_count = 0
            
            # 显示帧
            if not args.no_display:
                # 调整显示窗口大小
                display_frame = frame.copy()
                max_display_size = 1280
                if width > max_display_size or height > max_display_size:
                    scale = max_display_size / max(width, height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))
                
                cv2.imshow('Pig Detection - Video Tracking', display_frame)
                
                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # 退出
                    print("\n🛑 用户中断!")
                    break
                elif key == ord('p'):  # 暂停/继续
                    paused = not paused
                    print(f"  状态: {'已暂停' if paused else '继续'}")
                elif key == ord('s'):  # 保存当前帧
                    frame_path = os.path.join(args.out_dir, f'frame_{frame_count:06d}.jpg')
                    cv2.imwrite(frame_path, frame)
                    print(f"  已保存当前帧: {frame_path}")
                elif key == ord(' '):  # 空格键单步前进
                    paused = True
                    print("  单步模式 - 按空格继续下一帧")
            
            # 打印进度
            if frame_count % 30 == 0 and not paused:
                if total_frames > 0:
                    progress = (frame_count / total_frames * 100)
                    print(f"  进度: 第{frame_count}帧 ({progress:.1f}%) | 检测目标: {total_detections} | 跟踪目标: {total_tracks} | 实时FPS: {display_fps:.1f}")
                else:
                    print(f"  进度: 第{frame_count}帧 | 检测目标: {total_detections} | 跟踪目标: {total_tracks} | 实时FPS: {display_fps:.1f}")
    
    except KeyboardInterrupt:
        print("\n🛑 推理被用户中断!")
    except Exception as e:
        print(f"\n❌ 推理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # ============ 保存跟踪结果 ============
        if args.save_tracks and all_tracks:
            tracks_file = os.path.join(tracks_dir, 'tracks.json')
            import json
            with open(tracks_file, 'w') as f:
                json.dump(all_tracks, f, indent=2, ensure_ascii=False)
            print(f"📝 跟踪结果保存至: {tracks_file}")
            
            # 同时保存为CSV格式
            import csv
            csv_file = os.path.join(tracks_dir, 'tracks.csv')
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['frame', 'track_id', 'class_id', 'class_name', 'confidence', 'bbox'])
                for track in all_tracks:
                    writer.writerow([
                        track['frame'],
                        track['track_id'],
                        track['class_id'],
                        track['class_name'],
                        track['confidence'],
                        str(track['bbox'])
                    ])
            print(f"📊 跟踪CSV保存至: {csv_file}")
        
        # ============ 清理资源 ============
        if not args.no_display:
            cv2.destroyAllWindows()
        
        cap.release()
        if out_video:
            out_video.release()
        
        # ============ 打印统计信息 ============
        print("\n" + "=" * 60)
        print("📊 视频推理与跟踪统计")
        print("=" * 60)
        print(f"总帧数: {frame_count}")
        print(f"处理帧数: {processed_frames}")
        print(f"总检测目标数: {total_detections}")
        print(f"总跟踪目标数: {total_tracks}")
        print(f"最大跟踪ID: 需要从跟踪器获取")
        
        if processed_frames > 0:
            print(f"平均每帧检测数: {total_detections/processed_frames:.2f}")
            print(f"平均每帧跟踪数: {total_tracks/processed_frames:.2f}")
            print(f"总推理时间: {total_inference_time:.2f}s")
            print(f"平均每帧推理时间: {total_inference_time/processed_frames*1000:.1f}ms")
            print(f"平均处理FPS: {processed_frames/total_inference_time:.2f}")
        
        if args.save_video and out_video:
            print(f"输出视频保存至: {output_path}")
        
        if args.save_txt:
            print(f"检测结果保存至: {txt_dir}")
        
        if args.save_tracks:
            print(f"跟踪结果保存至: {tracks_dir}")
        
        print(f"所有结果保存至: {args.out_dir}")

if __name__ == '__main__':
    args = parse_args()
    main(args)