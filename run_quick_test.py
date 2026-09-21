import os
import subprocess
import time
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# 注意：这里我们使用 8, 9, 10, 13, 14 几个视频进行极速测试
BASE_CMD = 'python tools/track_test/evaluate_tracking_selected.py --base_dir "E:/pigtrackvideo" --video_ids 8 9 10 13 14 --max_age 60 --iou_threshold 0.8'

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

    return {
        "Exp": exp["name"], "Cost": cost_t, "Output": output,
        "MOTA": mota.group(1) if mota else "N/A",
        "IDF1": idf1.group(1) if idf1 else "N/A"
    }

def main():
    # 🌟 彻底抛弃 OAMM，引入我们纯正的 SGMP 进行消融！
    # experiments = [
    #     {"name": "01_Baseline",                         "args": ""},
    #     {"name": "03_SGMP_Only (New Core)",             "args": "--use_sgmp"}, 
    #     {"name": "06_BEV_plus_FQA (Current SOTA)",      "args": "--use_bev --use_deform"},
    #     {"name": "08_Ours_Full_SOTA (The Ultimate)",    "args": "--use_bev --use_deform --use_sgmp"} 
    # ]
    experiments = [
        # 1. 被剥去外挂的纯净卡尔曼滤波（你会看到它的真实水平其实很差）
        {"name": "01_Raw_Baseline",                     "args": ""},
        
        # 2. 不用外挂，单靠 SGMP 纯物理法则，看它能不能在线稳住轨迹
        {"name": "02_Baseline_plus_SGMP",               "args": "--use_sgmp"}, 
        
        # 3. 三大在线算法合体：空间(BEV) + 外观(FQA) + 运动(SGMP)
        {"name": "03_All_Online_Modules",               "args": "--use_bev --use_deform --use_sgmp"},
        
        # 4. 终极体：三大算法基底 + 你的全局离线缝合外挂 (碾压局)
        {"name": "04_Ours_Ultimate_SOTA",               "args": "--use_bev --use_deform --use_sgmp --use_postproc"} 
    ]

    print("="*80)
    print("🚀 启动极速冒烟测试 (验证 SGMP 纯物理约束模型)...")
    print("="*80)
    
    results_summary = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(run_exp, exp): exp for exp in experiments}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            print(f"[{i+1:02d}/4] 完成: {res['Exp']:<32} (耗时: {res['Cost']:>4.1f}s) | MOTA: {res['MOTA']}")
            results_summary.append(res)

    # 排序并打印简易表格
    results_summary.sort(key=lambda x: x["Exp"])
    print("\n" + "="*50)
    print(f"{'Experiment':<34} | {'MOTA':<6} | {'IDF1':<6}")
    print("-" * 50)
    for res in results_summary:
        print(f"{res['Exp']:<34} | {res['MOTA']:<6} | {res['IDF1']:<6}")
    print("="*50)

if __name__ == "__main__":
    main()