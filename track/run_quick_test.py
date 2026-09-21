import os
import subprocess
import time
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# 🌟 极速测试修改 1：只跑 1、2、3 三个视频！速度提升 5 倍以上
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
    idsw = re.search(r'IDSW:\s*([0-9.-]+)', output)
    motp = re.search(r'MOTP:\s*([0-9.-]+)', output)

    fps_matches = re.findall(r'Speed:\s*([0-9.]+)\s*FPS', output)
    fps_str = f"{sum(float(x) for x in fps_matches) / len(fps_matches):.1f}" if fps_matches else "N/A"

    return {
        "Exp": exp["name"], "Cost": cost_t, "Output": output,
        "MOTA": mota.group(1) if mota else "N/A",
        "IDF1": idf1.group(1) if idf1 else "N/A",
        "IDSW": idsw.group(1) if idsw else "N/A"
    }

def main():
    # 🌟 极速测试修改 2：只跑最核心的 3 组，验证趋势反转！
    # experiments = [
    #     {"name": "01_Baseline",                "args": ""},
    #     {"name": "02_FQA_Gate_Only",           "args": "--use_deform"}, 
    #     {"name": "08_Ours_Full_SOTA",          "args": "--use_bev --use_oamm --use_deform"} 
    # ]
    # experiments = [
    #     {"name": "01_Baseline",                      "args": ""},
    #     {"name": "02_FQA_Gate_Only",                 "args": "--use_deform"}, 
    #     {"name": "03_OAKF_Only",                     "args": "--use_oamm"}, 
    #     {"name": "06_BEV_plus_FQA",                  "args": "--use_bev --use_deform"} 
    # ]
    experiments = [
        {"name": "01_Baseline",                         "args": ""},
        {"name": "03_Bio_OAMM_Only",                    "args": "--use_oamm"}, 
        {"name": "06_BEV_plus_FQA (Current SOTA)",      "args": "--use_bev --use_deform"},
        {"name": "08_Ours_Full_SOTA (New Challenger)",  "args": "--use_bev --use_deform --use_oamm"} 
    ]

    print("="*80)
    print("启动极速冒烟测试 (只跑 Video 8,9,10,13,14 ...")
    print("="*80)
    
    results_summary = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(run_exp, exp): exp for exp in experiments}
        for i, future in enumerate(as_completed(futures)):
            res = future.result()
            print(f"[{i+1:02d}/4] 完成: {res['Exp']:<20} (耗时: {res['Cost']:>4.1f}s) | MOTA: {res['MOTA']}")
            results_summary.append(res)

    # 排序并打印简易表格
    results_summary.sort(key=lambda x: x["Exp"])
    print("\n" + "="*50)
    print(f"{'Experiment':<22} | {'MOTA':<6} | {'IDF1':<6}")
    print("-" * 50)
    for res in results_summary:
        print(f"{res['Exp']:<22} | {res['MOTA']:<6} | {res['IDF1']:<6}")
    print("="*50)

if __name__ == "__main__":
    main()