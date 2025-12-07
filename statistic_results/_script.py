import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def moving_average(y, window=3):
    return np.convolve(y, np.ones(window) / window, mode='valid')


PREFIX = 'statistic_results'

cn_files = [
    # 'results_steps_cn.csv',
    'results_time_cn.csv',
    'results_vms_cn.csv',
    'results_jp_cn.csv'
]

mc_files = [
    # 'results_steps_mc.csv',
    'results_time_mc.csv',
    'results_vms_mc.csv',
    'results_jp_mc.csv'
]

file_label_mapper = {
    'steps': r"$p$",
    'time': r"TIME(ms)",
    'vms': r"$|\mathcal{P}_v|$",
    'jp': r"$|Calls|$",
}


def draw(data, xlabel, ylabel):
    for file in data:
        df = pd.read_csv(file)

        x = df["number"]
        y = df["value"]

        # y_smooth = savgol_filter(y, window_length=5, polyorder=2)

        y_smooth = moving_average(y, window=3)
        x_smooth = x[len(x) - len(y_smooth):]
        for keyword, label in file_label_mapper.items():
            if keyword in file:
                plt.plot(x_smooth, y_smooth, label=label)
                break

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()


draw(cn_files, xlabel='minimum number', ylabel='value')
draw(mc_files, xlabel='number of children', ylabel='value')