from pathlib import Path
from utils.utils import plot_line_chart, min_mse_average, excel_to_numpy, centered_moving_window_variance
import tqdm
import argparse
import numpy as np


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


def get_args():
    parser = argparse.ArgumentParser(description='one dim Subdomain filtering')
    parser.add_argument('--epoch', type=int, default=5, help='迭代次数')
    parser.add_argument('--file_path', type=str, default=None, help='重力异常文件地址,目前支持xlsx文件')
    parser.add_argument('--subdomain_size', type=int, default=5, help="子域大小,只能为奇数")
    parser.add_argument('--line', type=str, default="row", help="row 行 / column 列")
    parser.add_argument('--num', type=int, default=80, help="取用的行数或者列数")
    # parser.add_argument('--processes', type=int, default=None, help='并行进程数，默认使用CPU核心数')
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = get_args()
    epoch = args.epoch
    file_path = args.file_path
    size = args.subdomain_size
    num = args.num
    line = args.line

    if file_path is not None:
        if Path(file_path).suffix == ".xlsx":
            if line == "row":
                data = excel_to_numpy(file_path)[num, :].tolist()
            elif line == "column":
                data = excel_to_numpy(file_path)[:, num].tolist()
        else:
            raise ValueError("暂时不支持其他格式文件")
    else:
        data = [60, 60, 60, 60, 60, 59, 57, 55, 52, 46, 36, 26, 20, 17, 15, 13, 12, 12, 12, 12, 12]

    size = 5
    epoch = 5
    plot_line_chart(data, show_labels=False, show=True)
    result = [data]
    for i in tqdm.tqdm(range(epoch)):
        data = centered_moving_window_variance(data, size)
        result.append(data)
    plot_line_chart(result, show_labels=False, show=True)


