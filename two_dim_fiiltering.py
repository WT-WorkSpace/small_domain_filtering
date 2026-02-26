import argparse
import tqdm
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
        min_mean, min_msd = min_mse_average(
            [win[:radius + 1], win[window_size - radius - 1:]]
        )
        res.append(min_mean)

    return res


def get_args():
    parser = argparse.ArgumentParser(description='multi-direction 1D Subdomain filtering')
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--file_path', type=str, default=r"D:\Code\small_domain_filtering\data\gong.grd")
    parser.add_argument('--subdomain_size', type=int, default=5)
    parser.add_argument('--output', type=str, default="output")
    parser.add_argument('--plot_levels', type=int, default=50)
    parser.add_argument('--plot_type', type=str, default="filled")
    parser.add_argument('--vis', type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()
    matrix = grd_to_numpy(args.file_path)
    print("矩阵大小:", matrix.shape)

    time = get_current_date_formatted()
    stem = Path(args.file_path).stem
    output_path = os.path.join(args.output, stem, time)
    output_path_png = os.path.join(output_path, "png")
    mkdir_if_not_exist(output_path_png)

    for it in range(args.epoch):
        print(f"\n------ 第 {it + 1} 轮迭代 ------")

        h, w = matrix.shape

        # 0° 行方向
        row_result = np.array([
            centered_moving_window_variance(matrix[i, :], args.subdomain_size)
            for i in range(h)
        ])

        # 90° 列方向
        col_result = np.array([
            centered_moving_window_variance(matrix[:, j], args.subdomain_size)
            for j in range(w)
        ]).T

        # 45° 主对角线方向
        diag45 = np.zeros_like(matrix)
        count45 = np.zeros_like(matrix)

        for k in range(-(h - 1), w):
            coords = [(i, i + k) for i in range(h) if 0 <= i + k < w]
            if len(coords) < args.subdomain_size:
                continue
            data = [matrix[i, j] for i, j in coords]
            filtered = centered_moving_window_variance(data, args.subdomain_size)
            for (i, j), v in zip(coords, filtered):
                diag45[i, j] += v
                count45[i, j] += 1

        diag45 /= np.maximum(count45, 1)

        # 135° 副对角线方向
        diag135 = np.zeros_like(matrix)
        count135 = np.zeros_like(matrix)

        for k in range(h + w):
            coords = [(i, k - i) for i in range(h) if 0 <= k - i < w]
            if len(coords) < args.subdomain_size:
                continue
            data = [matrix[i, j] for i, j in coords]
            filtered = centered_moving_window_variance(data, args.subdomain_size)
            for (i, j), v in zip(coords, filtered):
                diag135[i, j] += v
                count135[i, j] += 1

        diag135 /= np.maximum(count135, 1)

        # 四方向平均
        matrix = (row_result + col_result + diag45 + diag135) / 4.0

        plot_contour(
            matrix,
            levels=args.plot_levels,
            title=f"iter_{it + 1}_multi_dir",
            plot_type=args.plot_type,
            save_path=os.path.join(output_path_png, f"iter_{it + 1}.png"),
            show_plot=args.vis
        )

    print("\n✔ 多方向一维子域滤波完成")
