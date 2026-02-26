import argparse
import numpy as np
import math
from pathlib import Path

from utils.utils import *
from utils.grid_clip import *


def extrapolate_data(data, n=2):
    data = np.array(data)
    left_x = np.arange(-n, 0)
    left_slope = data[1] - data[0]
    left_extrapolation = data[0] + left_slope * (left_x + 1)

    right_x = np.arange(1, n + 1)
    right_slope = data[-1] - data[-2]
    right_extrapolation = data[-1] + right_slope * right_x

    return np.concatenate([left_extrapolation, data, right_extrapolation])


def centered_moving_window_variance(data, window_size):
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("窗口大小必须为正奇数")

    radius = (window_size - 1) // 2
    data = extrapolate_data(data, radius)

    res = []
    windows = np.lib.stride_tricks.sliding_window_view(data, window_size)

    for win in windows:
        min_mean, _ = min_mse_average(
            [win[:radius + 1], win[window_size - radius - 1:]]
        )
        res.append(min_mean)

    return res


def get_args():
    parser = argparse.ArgumentParser("Multi-angle 1D subdomain filtering")
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--file_path', type=str, default=r"D:\Code\small_domain_filtering\data\gong.grd")
    parser.add_argument('--subdomain_size', type=int, default=5)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--plot_levels', type=int, default=50)
    parser.add_argument('--plot_type', type=str, default='filled')
    parser.add_argument('--vis', type=bool, default=False)
    return parser.parse_args()


def generate_lines(matrix_shape, angle_deg):
    """
    生成给定角度下的多条离散直线（像素坐标）
    """
    h, w = matrix_shape
    theta = math.radians(angle_deg)
    dx = math.cos(theta)
    dy = math.sin(theta)

    lines = []

    # 从边界出发
    for y0 in range(h):
        x0 = 0
        coords = []
        x, y = x0, y0
        while 0 <= int(x) < w and 0 <= int(y) < h:
            coords.append((int(y), int(x)))
            x += dx
            y += dy
        if len(coords) > 1:
            lines.append(coords)

    for x0 in range(w):
        y0 = 0
        coords = []
        x, y = x0, y0
        while 0 <= int(x) < w and 0 <= int(y) < h:
            coords.append((int(y), int(x)))
            x += dx
            y += dy
        if len(coords) > 1:
            lines.append(coords)

    return lines


if __name__ == "__main__":

    args = get_args()

    if Path(args.file_path).suffix == ".grd":
        matrix = grd_to_numpy(args.file_path)
    elif Path(args.file_path).suffix == ".npy":
        matrix = np.load(args.file_path)
    else:
        raise ValueError("不支持的文件格式")

    print("矩阵大小:", matrix.shape)

    angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

    time_str = get_current_date_formatted()
    stem = Path(args.file_path).stem
    out_dir = Path(args.output) / stem / time_str
    png_dir = out_dir / "png"
    mkdir_if_not_exist(png_dir)

    for it in range(args.epoch):
        print(f"\n------ 第 {it + 1} 轮迭代 ------")

        acc = np.zeros_like(matrix, dtype=float)
        cnt = np.zeros_like(matrix, dtype=int)

        for angle in angles:
            print(f"  -> 方向 {angle}°")
            lines = generate_lines(matrix.shape, angle)

            for line in lines:
                if len(line) < args.subdomain_size:
                    continue

                data = [matrix[i, j] for i, j in line]
                filtered = centered_moving_window_variance(
                    data, args.subdomain_size
                )

                for (i, j), v in zip(line, filtered):
                    acc[i, j] += v
                    cnt[i, j] += 1

        matrix = acc / np.maximum(cnt, 1)

        plot_contour(
            matrix,
            levels=args.plot_levels,
            title=f"iter_{it + 1}_multi_angle",
            plot_type=args.plot_type,
            save_path=str(png_dir / f"iter_{it + 1}.png"),
            show_plot=args.vis
        )

    print("\n✔ 任意角度一维子域滤波完成")
