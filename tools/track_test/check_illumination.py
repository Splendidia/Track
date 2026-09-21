import os
import glob
import json
import numpy as np
import cv2

def compute_illumination_score(patch_bgr):
    """
    计算图像块的光照与清晰度质量分数，返回 0.0 ~ 1.0
    """
    if patch_bgr is None or patch_bgr.size == 0 or patch_bgr.shape[0] == 0 or patch_bgr.shape[1] == 0:
        return 0.0
    
    # 1. 转为灰度图
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. 亮度得分 (均值除以255)
    mean_brightness = np.mean(gray) / 255.0
    
    # 3. 对比度/纹理得分 (拉普拉斯方差)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 假设方差500是非常清晰的阈值，进行截断归一化
    contrast_score = min(laplacian_var / 500.0, 1.0) 
    
    # 4. 融合得分 (亮度与对比度各占50%)
    iqe_score = 0.5 * mean_brightness + 0.5 * contrast_score
    
    # 限制在 0.1 ~ 0.9 之间
    return np.clip(iqe_score, 0.1, 0.9)

def load_gt_frames(video_folder):
    """提取真实标注框 (8点多边形)"""
    frames = {}
    json_files = sorted(glob.glob(os.path.join(video_folder, '*.json')))
    for jf in json_files:
        with open(jf) as f: 
            data = json.load(f)
        frame_id = int(data['imagePath'].split('_')[1].split('.')[0])
        boxes = []
        for shape in data['shapes']:
            boxes.append(np.array(shape['points']).flatten().tolist())
        frames[frame_id] = boxes
    return frames

def main():
    # 假设我们拿视频 10 来做验证（你可以改成其他光照变化明显的视频）
    video_id = "14"
    base_dir = r"E:\pigtrackvideo"
    video_folder = os.path.join(base_dir, video_id)
    
    if not os.path.exists(video_folder):
        print(f"找不到视频路径: {video_folder}")
        return

    print(f"启动暗区感知器验证: {video_folder}")
    gt_frames = load_gt_frames(video_folder)
    
    # 设置显示窗口
    cv2.namedWindow("Illumination Gate Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Illumination Gate Test", 1280, 720)

    for frame_id in sorted(gt_frames.keys()):
        img_path = os.path.join(video_folder, f'frame_{frame_id:05d}.jpg')
        if not os.path.exists(img_path): continue
        
        img = cv2.imread(img_path)
        boxes = gt_frames.get(frame_id, [])
        
        for coords in boxes:
            # 1. 解析 8 个点的坐标，获取水平外接矩形
            pts = np.array(coords).reshape(4, 2).astype(np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            
            # 边界保护
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
            
            # 2. 抠出猪体的 Patch
            patch = img[y1:y2, x1:x2]
            
            # 3. 计算光照质量分数
            score = compute_illumination_score(patch)
            
            # 4. 可视化：亮区(高分)标绿色，暗区(低分)标红色
            color = (0, 255, 0) if score >= 0.5 else (0, 0, 255)
            thickness = 2 if score >= 0.5 else 4 # 暗区加粗显示，方便观察
            
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
            
            # 在框的左上角打印分数
            label = f"Score: {score:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # 显示图片
        cv2.imshow("Illumination Gate Test", img)
        
        # 按 'q' 键退出，按空格键暂停
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("验证结束！")

if __name__ == '__main__':
    main()