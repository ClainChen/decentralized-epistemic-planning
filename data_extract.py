import os
import re
import csv

results_steps_cn = []
results_time_cn = []
results_vms_cn = []
results_jp_cn = []

results_steps_mc = []
results_time_mc = []
results_vms_mc = []
results_jp_mc = []

PREFIX = "statistic_results"

mapper = {
    'results_steps_cn.csv': results_steps_cn,
    'results_time_cn.csv': results_time_cn,
    'results_vms_cn.csv': results_vms_cn,
    'results_jp_cn.csv': results_jp_cn,
    'results_steps_mc.csv': results_steps_mc,
    'results_time_mc.csv': results_time_mc,
    'results_vms_mc.csv': results_vms_mc,
    'results_jp_mc.csv': results_jp_mc,
}

for file in os.listdir('out'):
    if not file.endswith('.out'):
        continue

    if not file.startswith('consecutive_number') and not file.startswith('muddy_children'):
        continue

    filepath = os.path.join('out', file)
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    last_four = lines[-5:]

    values = []
    for line in last_four:
        match = re.findall(r"(\d+\.\d+)", line)
        if match:
            number = float(match[0])
            values.append(number)
    avg_steps, avg_time, avg_vms, avg_jp = values

    formatted = [
        round(avg_steps, 1),
        round(avg_time * 1000, 3),
        round(avg_vms, 1),
        round(avg_jp, 1)
    ]

    if file.startswith('consecutive_number'):
        id = int(re.findall(r"consecutive_number-(\d+).out", file)[0])

        results_steps_cn.append([id, round(avg_steps, 1)])
        results_time_cn.append([id, round(avg_time * 1000, 1)])
        results_vms_cn.append([id, round(avg_vms, 1)])
        results_jp_cn.append([id, round(avg_jp, 1)])
    else:
        id = int(re.findall(r"muddy_children-muddy_(\d+).out", file)[0])

        results_steps_mc.append([id, round(avg_steps, 1)])
        results_time_mc.append([id, round(avg_time * 1000, 1)])
        results_vms_mc.append([id, round(avg_vms, 1)])
        results_jp_mc.append([id, round(avg_jp, 1)])

for file_name, lst in mapper.items():
    with open(os.path.join(PREFIX,file_name), "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['number', 'value'])
        writer.writerows(sorted(lst, key=lambda x: x[0]))
    print(f"Finish output：{file_name}")
