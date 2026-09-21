# import os
# import subprocess
# import time
# import re
# from concurrent.futures import ProcessPoolExecutor, as_completed


# BASE_CMD = 'python tools/track_test/evaluate_tracking_selected.py --base_dir "E:/pigtrackvideo" --video_ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --max_age 60 --iou_threshold 0.8'

# def run_exp(exp):
#     cmd = f'{BASE_CMD} {exp["args"]} --exp_name "{exp["name"]}"'
#     start_t = time.time()
    
#     env = os.environ.copy()
#     env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    
#     process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='gbk', errors='ignore', env=env)
#     output, _ = process.communicate()
#     cost_t = time.time() - start_t

#     mota = re.search(r'MOTA:\s*([0-9.-]+)', output)
#     idf1 = re.search(r'IDF1:\s*([0-9.-]+)', output)
#     idsw = re.search(r'IDSW:\s*([0-9.-]+)', output)
#     motp = re.search(r'MOTP:\s*([0-9.-]+)', output)

#     fps_matches = re.findall(r'Speed:\s*([0-9.]+)\s*FPS', output)
#     if fps_matches:
#         avg_fps = sum(float(x) for x in fps_matches) / len(fps_matches)
#         fps_str = f"{avg_fps:.1f}"
#     else:
#         fps_str = "N/A"

#     return {
#         "Exp": exp["name"], "Cost": cost_t, "Output": output,
#         "MOTA": mota.group(1) if mota else "N/A",
#         "IDF1": idf1.group(1) if idf1 else "N/A",
#         "IDSW": idsw.group(1) if idsw else "N/A",
#         "MOTP": motp.group(1) if motp else "N/A",
#         "FPS": fps_str  
#     }

# def main():
#     # 极致纯粹的 7 组消融实验（聚焦: BEV空间 + OAKF运动 + FQA表观门控）
#     experiments = [
#         {"name": "01_Baseline",                "args": ""},
#         {"name": "02_FQA_Gate_Only",           "args": "--use_deform"}, 
#         {"name": "03_OAKF_Only",               "args": "--use_oamm"},   
#         {"name": "04_BEV_Only",                "args": "--use_bev"},    
#         {"name": "05_BEV_plus_FQA_Gate",       "args": "--use_bev --use_deform"},
#         {"name": "06_BEV_plus_OAKF",           "args": "--use_bev --use_oamm"},
#         {"name": "07_Ours_Full_SOTA",          "args": "--use_bev --use_oamm --use_deform"} 
#     ]

#     log_file = "paper_ablation_results_core3.txt"
#     results_summary = []

#     print("="*90)
#     print("启动核心三大模块消融实验 (IoU 0.8 黄金门限：冲刺完美论文大表)...")
#     print("="*90)
#     total_start_time = time.time()

#     with open(log_file, "w", encoding="utf-8") as f:
#         f.write("=== Final Ablation Study Execution Log (Core 3 Modules) ===\n\n")

#     with ProcessPoolExecutor(max_workers=3) as executor:
#         futures = {executor.submit(run_exp, exp): exp for exp in experiments}
#         for i, future in enumerate(as_completed(futures)):
#             res = future.result()
#             print(f"[{i+1:02d}/{len(experiments)}] 完成: {res['Exp']:<24} (耗时: {res['Cost']:>5.1f}s) | MOTA: {res['MOTA']}")
#             results_summary.append(res)
            
#             with open(log_file, "a", encoding="utf-8") as f:
#                 f.write(f"--- Output for {res['Exp']} ---\n{res['Output']}\n\n")

#     results_summary.sort(key=lambda x: x["Exp"])

#     table_divider = "=" * 85
#     title = "FINAL ABLATION STUDY - CORE 3 MODULES (IoU Threshold: 0.8)"
#     header = f"{'Experiment':<24} | {'MOTA':<6} | {'IDF1':<6} | {'IDSW':<6} | {'MOTP':<6} | {'FPS':<6}"
    
#     print("\n" + table_divider)
#     print(title)
#     print(table_divider)
#     print(header)
#     print("-" * 85)

#     table_str = f"{table_divider}\n{title}\n{table_divider}\n{header}\n{'-' * 85}\n"

#     for res in results_summary:
#         row = f"{res['Exp']:<24} | {res['MOTA']:<6} | {res['IDF1']:<6} | {res['IDSW']:<6} | {res['MOTP']:<6} | {res['FPS']:<6}"
#         print(row)
#         table_str += row + "\n"

#     print(table_divider)
#     total_cost = (time.time() - total_start_time) / 60
#     summary_text = f"所有实验执行完毕！总耗时: {total_cost:.1f} 分钟。结果已保存至 {log_file}"
#     print(summary_text)

#     with open(log_file, "a", encoding="utf-8") as f:
#         f.write("\n" + table_str + summary_text + "\n")

# if __name__ == "__main__":
#     main()

import os
import subprocess
import time
import re
from concurrent.futures import ProcessPoolExecutor, as_completed


BASE_CMD = 'python tools/track_test/evaluate_tracking_selected.py --base_dir "E:/pigtrackvideo" --video_ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --max_age 60 --iou_threshold 0.8'
# BASE_CMD = 'python tools/track_test/evaluate_tracking_selected.py --base_dir "E:/pigtrackvideo" --video_ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --max_age 30 --iou_threshold 0.5'
def run_exp(exp):
    cmd = f'{BASE_CMD} {exp["args"]} --exp_name "{exp["name"]}"'
    start_t = time.time()
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='gbk', errors='ignore', env=env)
    output, _ = process.communicate()
    cost_t = time.time() - start_t

    mota = re.search(r'MOTA:\s*([0-9.-]+)', output)
    idf1 = re.search(r'IDF1:\s*([0-9.-]+)', output)
    idsw = re.search(r'IDSW:\s*([0-9.-]+)', output)
    motp = re.search(r'MOTP:\s*([0-9.-]+)', output)

    fps_matches = re.findall(r'Speed:\s*([0-9.]+)\s*FPS', output)
    if fps_matches:
        avg_fps = sum(float(x) for x in fps_matches) / len(fps_matches)
        fps_str = f"{avg_fps:.1f}"
    else:
        fps_str = "N/A"

    return {
        "Exp": exp["name"], "Cost": cost_t, "Output": output,
        "MOTA": mota.group(1) if mota else "N/A",
        "IDF1": idf1.group(1) if idf1 else "N/A",
        "IDSW": idsw.group(1) if idsw else "N/A",
        "MOTP": motp.group(1) if motp else "N/A",
        "FPS": fps_str  
    }

def main():
    # 极致纯粹的 8 组消融实验（聚焦: BEV空间 + OAKF运动 + FQA表观门控，完美2^3全排列）
    experiments = [
        {"name": "01_Baseline",                "args": ""},
        {"name": "02_FQA_Gate_Only",           "args": "--use_deform"}, 
        {"name": "03_RIFM_Only",               "args": "--use_rifm"}, 
        # {"name": "03_OAKF_Only",               "args": "--use_oamm"},   
        {"name": "04_BEV_Only",                "args": "--use_bev"},    
        {"name": "05_FQA_plus_RIFM",           "args": "--use_deform --use_rifm"}, 
        # {"name": "05_FQA_plus_OAKF",           "args": "--use_deform --use_oamm"}, 
        {"name": "06_BEV_plus_FQA_Gate",       "args": "--use_bev --use_deform"},
        {"name": "07_BEV_plus_RIFM",           "args": "--use_bev --use_rifm"},
        # {"name": "07_BEV_plus_OAKF",           "args": "--use_bev --use_oamm"},
        {"name": "08_Ours_Full_SOTA",          "args": "--use_bev --use_rifm --use_deform"}
        # {"name": "08_Ours_Full_SOTA",          "args": "--use_bev --use_oamm --use_deform"},
        # {"name": "09_SGMP_Only", "args": "--use_sgmp"}
    ]

    log_file = "paper_ablation_results_core3.txt"
    results_summary = []

    print("="*90)
    print("启动核心三大模块消融实验 (IoU 0.8 黄金门限：冲刺完美论文大表 - 8组全排列)...")
    print("="*90)
    total_start_time = time.time()

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== Final Ablation Study Execution Log (Core 3 Modules - 8 Exps) ===\n\n")

    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_exp, exp): exp for exp in experiments}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            print(f"[{i+1:02d}/{len(experiments)}] 完成: {res['Exp']:<24} (耗时: {res['Cost']:>5.1f}s) | MOTA: {res['MOTA']}")
            results_summary.append(res)
            
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"--- Output for {res['Exp']} ---\n{res['Output']}\n\n")

    results_summary.sort(key=lambda x: x["Exp"])

    table_divider = "=" * 85
    title = "FINAL ABLATION STUDY - CORE 3 MODULES (IoU Threshold: 0.8)"
    header = f"{'Experiment':<24} | {'MOTA':<6} | {'IDF1':<6} | {'IDSW':<6} | {'MOTP':<6} | {'FPS':<6}"
    
    print("\n" + table_divider)
    print(title)
    print(table_divider)
    print(header)
    print("-" * 85)

    table_str = f"{table_divider}\n{title}\n{table_divider}\n{header}\n{'-' * 85}\n"

    for res in results_summary:
        row = f"{res['Exp']:<24} | {res['MOTA']:<6} | {res['IDF1']:<6} | {res['IDSW']:<6} | {res['MOTP']:<6} | {res['FPS']:<6}"
        print(row)
        table_str += row + "\n"

    print(table_divider)
    total_cost = (time.time() - total_start_time) / 60
    summary_text = f"所有实验执行完毕！总耗时: {total_cost:.1f} 分钟。结果已保存至 {log_file}"
    print(summary_text)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n" + table_str + summary_text + "\n")

if __name__ == "__main__":
    main()

# import os
# import subprocess
# import time
# import re
# from concurrent.futures import ProcessPoolExecutor, as_completed

# # 【修改点 1】选取多个视频（这里暂定 1 2 3，代表多视频验证集）
# # 【修改点 2】采用学术标准 IoU 0.5
# BASE_CMD = 'python tools/track_test/evaluate_tracking_selected.py --base_dir "E:/pigtrackvideo" --video_ids 8 9 10 13 14 --max_age 60 --iou_threshold 0.5'

# # 当你验证成功后，可以直接把上面那行注释掉，换成下面这行全量 14 个视频的：
# # BASE_CMD = 'python tools/track_test/evaluate_tracking_selected.py --base_dir "E:/pigtrackvideo" --video_ids 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --max_age 60 --iou_threshold 0.5'

# def run_exp(exp):
#     cmd = f'{BASE_CMD} {exp["args"]} --exp_name "{exp["name"]}"'
#     start_t = time.time()
    
#     env = os.environ.copy()
#     env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
    
#     process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='gbk', errors='ignore', env=env)
#     output, _ = process.communicate()
#     cost_t = time.time() - start_t

#     mota = re.search(r'MOTA:\s*([0-9.-]+)', output)
#     idf1 = re.search(r'IDF1:\s*([0-9.-]+)', output)
#     idsw = re.search(r'IDSW:\s*([0-9.-]+)', output)
#     motp = re.search(r'MOTP:\s*([0-9.-]+)', output)

#     fps_matches = re.findall(r'Speed:\s*([0-9.]+)\s*FPS', output)
#     if fps_matches:
#         avg_fps = sum(float(x) for x in fps_matches) / len(fps_matches)
#         fps_str = f"{avg_fps:.1f}"
#     else:
#         fps_str = "N/A"

#     return {
#         "Exp": exp["name"], "Cost": cost_t, "Output": output,
#         "MOTA": mota.group(1) if mota else "N/A",
#         "IDF1": idf1.group(1) if idf1 else "N/A",
#         "IDSW": idsw.group(1) if idsw else "N/A",
#         "MOTP": motp.group(1) if motp else "N/A",
#         "FPS": fps_str  
#     }

# def main():
#     # 【修改点 3】只跑最关键的两个实验，验证趋势是否反转
#     experiments = [
#         {"name": "01_Baseline",                "args": ""},
#         {"name": "02_FQA_Gate_Only",           "args": "--use_deform"}, 
#         {"name": "08_Ours_Full_SOTA",          "args": "--use_bev --use_oamm --use_deform"} 
#         ]

#     log_file = "paper_ablation_results_quick_tune.txt"
#     results_summary = []

#     print("="*90)
#     print("🚀 启动多视频极速调参验证 (IoU 0.5: 验证 Full_SOTA 能否反超 FQA_Only)...")
#     print("="*90)
#     total_start_time = time.time()

#     with ProcessPoolExecutor(max_workers=2) as executor:
#         futures = {executor.submit(run_exp, exp): exp for exp in experiments}
#         for i, future in enumerate(as_completed(futures)):
#             res = future.result()
#             print(f"[{i+1:02d}/{len(experiments)}] 完成: {res['Exp']:<24} (耗时: {res['Cost']:>5.1f}s) | MOTA: {res['MOTA']} | IDSW: {res['IDSW']}")
#             results_summary.append(res)

#     results_summary.sort(key=lambda x: x["Exp"])

#     table_divider = "=" * 85
#     title = "QUICK TUNE CHECK - (IoU Threshold: 0.5, Multi-Video)"
#     header = f"{'Experiment':<24} | {'MOTA':<6} | {'IDF1':<6} | {'IDSW':<6} | {'MOTP':<6} | {'FPS':<6}"
    
#     print("\n" + table_divider)
#     print(title)
#     print(table_divider)
#     print(header)
#     print("-" * 85)

#     for res in results_summary:
#         row = f"{res['Exp']:<24} | {res['MOTA']:<6} | {res['IDF1']:<6} | {res['IDSW']:<6} | {res['MOTP']:<6} | {res['FPS']:<6}"
#         print(row)

#     print(table_divider)
    
#     # 自动判定逻辑
#     try:
#         mota_fqa = float(results_summary[0]['MOTA'])
#         mota_full = float(results_summary[1]['MOTA'])
#         if mota_full > mota_fqa:
#             print("\n✅ 恭喜！趋势完美！Full SOTA 的 MOTA 已经超越了纯 FQA。")
#             print("👉 下一步：你可以把脚本里的 BASE_CMD 换成全量 14 个视频，跑完整 8 组实验出最终表格了！")
#         else:
#             print("\n⚠️ 警告：Full SOTA 的 MOTA 依然落后。可能需要调小 OAKF 的噪声放大系数。")
#     except:
#         pass

# if __name__ == "__main__":
#     main()