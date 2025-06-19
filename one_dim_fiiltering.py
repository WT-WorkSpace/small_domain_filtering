import matplotlib.pyplot as plt
import numpy as np
from utils import plot_line_chart, min_mse_average
import tqdm
import copy
def extrapolate_data(data, n=2):
    data = np.array(data)
    left_x = np.arange(-n, 0)
    left_slope = data[1] - data[0]
    left_extrapolation = data[0] + left_slope * (left_x + 1)
    right_x = np.arange(1, n + 1)
    right_slope = data[-1] - data[-2]
    right_extrapolation = data[-1] + right_slope * right_x
    extended_data = np.concatenate([left_extrapolation, data, right_extrapolation])
    return extended_data
def centered_moving_window_variance(data, window_size):

    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("窗口大小必须为正奇数")

    radius = (window_size - 1) // 2
    data = extrapolate_data(data, radius)
    res = []
    windows = np.lib.stride_tricks.sliding_window_view(data, window_size)

    for win in windows:
        min_mean, min_msd = min_mse_average([win[:radius+1], win[window_size - radius-1:]])
        res.append(min_mean)

    return res

if __name__ == "__main__":
    data = [60, 60, 60, 60, 60, 59, 57, 55, 52, 46, 36, 26, 20, 17, 15, 13, 12, 12, 12, 12, 12]

    size = 5
    epoch = 5
    plot_line_chart(data, show=True)
    result = [data]
    for i in tqdm.tqdm(range(epoch)):
        data = np.round(centered_moving_window_variance(data, size),2)
        result.append(data)
    plot_line_chart(result, show=True)


