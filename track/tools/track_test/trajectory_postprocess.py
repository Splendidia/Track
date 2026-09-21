# vim: expandtab:ts=4:sw=4
import numpy as np
from collections import defaultdict
from scipy.interpolate import CubicSpline

class TrajectoryProcessor:
    """
    负责 SBS (缝合), GTF (剔除) 和 TBS (平滑) 的综合后处理模块。
    将离线轨迹优化的所有逻辑从 evaluate 脚本中彻底剥离。
    """
    
    @staticmethod
    def sbs_stitching(results_list, max_stitch_gap=60, max_interp_gap=30):
        """
        创新挂载：SBS (Search-Based Stitching) 启发式衰减全局缝合与三次样条插值
        """
        if not results_list: 
            return []
            
        tracklets = defaultdict(dict)
        for res in results_list:
            fid, tid = int(res[0]), int(res[1])
            tracklets[tid][fid] = res[2:7] 
            
        t_meta = {}
        for tid, frames in tracklets.items():
            f_list = sorted(frames.keys())
            t_meta[tid] = {
                'start_f': f_list[0], 'start_box': frames[f_list[0]],
                'end_f': f_list[-1], 'end_box': frames[f_list[-1]]
            }
            
        merged_to = {} 
        for tid2 in sorted(t_meta.keys()):
            best_tid1 = None
            best_gap = max_stitch_gap
            
            for tid1 in sorted(t_meta.keys()):
                if tid1 >= tid2: continue
                curr1 = tid1
                while curr1 in merged_to: curr1 = merged_to[curr1]
                if curr1 == tid2: continue 
                
                f1 = t_meta[curr1]['end_f']
                b1 = t_meta[curr1]['end_box']
                f2 = t_meta[tid2]['start_f']
                b2 = t_meta[tid2]['start_box']
                
                gap = f2 - f1
                if 0 < gap < best_gap:
                    c1x, c1y, w1, h1 = b1[:4]
                    c2x, c2y, w2, h2 = b2[:4]
                    
                    # 提取 BEV 脚印坐标进行物理距离计算
                    foot_1 = np.array([c1x, c1y + h1/2])
                    foot_2 = np.array([c2x, c2y + h2/2])
                    dist = np.sqrt((foot_1[0]-foot_2[0])**2 + ((foot_1[1]-foot_2[1])*2.5)**2)
                    
                    # SBS 核心逻辑：指数衰减搜索域
                    base_radius = max(w1, h1) * 1.5
                    # 随时间先对数扩张，后指数收缩，完美拟合生物运动学的不确定性
                    dynamic_radius = base_radius * (1.0 + np.log(gap + 1.0)) * np.exp(-0.015 * gap)
                    
                    if dist < dynamic_radius: 
                        best_tid1 = curr1
                        best_gap = gap
                        
            if best_tid1 is not None:
                merged_to[tid2] = best_tid1
                for f, b in tracklets[tid2].items():
                    tracklets[best_tid1][f] = b
                t_meta[best_tid1]['end_f'] = t_meta[tid2]['end_f']
                t_meta[best_tid1]['end_box'] = t_meta[tid2]['end_box']
                del tracklets[tid2]
                
        interpolated_results = []
        for tid, frames in tracklets.items():
            f_list = sorted(frames.keys())
            for i in range(len(f_list) - 1):
                f1, f2 = f_list[i], f_list[i+1]
                b1, b2 = np.array(frames[f1]), np.array(frames[f2])
                interpolated_results.append([f1, tid] + b1.tolist())
                
                gap = f2 - f1
                if 1 < gap <= max_interp_gap:
                    # ====================================================================
                    # 上下文感知的三次样条插值 (Kinematic Cubic Spline)
                    # ====================================================================
                    context_f = []
                    context_b = []
                    
                    # 确定上下文窗口：往前最多取3帧，往后最多取4帧 (含当前两端点)
                    start_idx = max(0, i - 3)
                    end_idx = min(len(f_list) - 1, i + 4)
                    
                    for j in range(start_idx, end_idx + 1):
                        fid_ctx = f_list[j]
                        box_ctx = np.array(frames[fid_ctx]).copy() # 必须copy防污染
                        
                        # 物理学绝招：角度连续化 (Unroll Angle)
                        if len(context_b) > 0:
                            prev_angle = context_b[-1][4]
                            curr_angle = box_ctx[4]
                            diff = (curr_angle - prev_angle + 90) % 180 - 90
                            box_ctx[4] = prev_angle + diff
                            
                        context_f.append(fid_ctx)
                        context_b.append(box_ctx)
                        
                    context_f = np.array(context_f)
                    context_b = np.array(context_b)

                    # 安全回退机制：点数不足以拟合平滑曲线时，回退到线性插值
                    if len(context_f) < 4 or len(np.unique(context_f)) != len(context_f):
                        if abs(b1[4] - b2[4]) > 90:
                            if b1[4] > 0: b2[4] += 180
                            else: b2[4] -= 180
                        for step in range(1, gap):
                            weight = step / gap
                            b_interp = b1 * (1 - weight) + b2 * weight
                            b_interp[4] = (b_interp[4] + 90) % 180 - 90 
                            interpolated_results.append([f1 + step, tid] + b_interp.tolist())
                    else:
                        try:
                            # 核心发力：拟合具有生物运动加减速惯性的 5 维曲线
                            cs = CubicSpline(context_f, context_b, axis=0, bc_type='natural')
                            for step in range(1, gap):
                                target_f = f1 + step
                                b_interp = cs(target_f)
                                # 重新将角度包裹回 [-90, 90] 的旋转框标准
                                b_interp[4] = (b_interp[4] + 90) % 180 - 90 
                                interpolated_results.append([target_f, tid] + b_interp.tolist())
                        except Exception:
                            # 极端异常情况回退
                            for step in range(1, gap):
                                weight = step / gap
                                b_interp = b1 * (1 - weight) + b2 * weight
                                b_interp[4] = (b_interp[4] + 90) % 180 - 90 
                                interpolated_results.append([f1 + step, tid] + b_interp.tolist())
                    # ====================================================================
                        
            if f_list:
                interpolated_results.append([f_list[-1], tid] + frames[f_list[-1]])
                
        return interpolated_results

    @staticmethod
    def apply_gtf_and_tbs(stitched_results, min_life=15, alpha=0.4):
        """
        GTF (Ghost Trajectory Filtering): 剔除存活时间过短的幽灵轨迹
        TBS (Temporal Bounding Box Smoothing): 用 EMA 消除边框的高频闪烁抖动
        """
        raw_pred = defaultdict(list)
        for res in stitched_results:
            fid, tid = int(res[0]), int(res[1])
            box = res[2:7]
            raw_pred[tid].append((fid, box))
            
        final_pred = {}
        for tid, frames in raw_pred.items():
            # GTF 过滤门控：如果一条轨迹从头到尾存活时间不到 15 帧，直接抹除
            if len(frames) < min_life:
                continue
                
            # 按帧号排序，准备进行时序平滑
            frames.sort(key=lambda x: x[0])
            
            # TBS 平滑：仅对 w(宽), h(高) 进行平滑，x, y 保持灵敏度
            smoothed_box = np.array(frames[0][1])
            
            for i in range(len(frames)):
                fid, current_box = frames[i][0], np.array(frames[i][1])
                
                smoothed_box[2:4] = alpha * current_box[2:4] + (1 - alpha) * smoothed_box[2:4]
                current_box[2:4] = smoothed_box[2:4]
                
                if fid not in final_pred:
                    final_pred[fid] = {'ids': [], 'boxes': []}
                final_pred[fid]['ids'].append(tid)
                final_pred[fid]['boxes'].append(current_box.tolist())
                
        return final_pred