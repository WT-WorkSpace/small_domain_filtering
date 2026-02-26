import argparse
import math
from utils.utils import *
from utils.grid_clip import *
import random

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
def plot_lines_only_random_color(matrix, lines, save_path=None, show=True):
    """
        绘制 generate_lines_angle 生成的线段，每条线随机颜色，并保存图像
        matrix: 2D array
        lines: list of [(y,x), ...]
        save_path: str 或 Path，保存图像的完整路径
        """
    matrix = np.array(matrix)
    if matrix.ndim != 2:
        raise ValueError("matrix 必须是二维数组")

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap='gray', origin='upper')

    for line in lines:
        if len(line) < 2:
            continue
        y_coords, x_coords = zip(*line)
        color = (random.random(), random.random(), random.random())
        plt.plot(x_coords, y_coords, linewidth=0.5, alpha=0.7, color=color)

    plt.axis('off')

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)  # 自动创建路径
    plt.savefig(str(save_path), bbox_inches='tight', dpi=200)
    plt.close()
    # print(f"✔ 图像已保存到 {save_path}")


def get_args():
    parser = argparse.ArgumentParser("Multi-angle 1D subdomain filtering with line visualization")
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--file_path', type=str,  default=r"D:\Code\small_domain_filtering\data\gong.grd")
    parser.add_argument('--subdomain_size', type=int, default=5)
    parser.add_argument('--output', type=str, default='output')
    parser.add_argument('--plot_levels', type=int, default=30)
    parser.add_argument('--plot_type', type=str, default='filled')
    parser.add_argument('--vis', type=bool, default=False)
    parser.add_argument('--show_lines', type=bool, default=True, help="是否绘制线段轨迹")
    return parser.parse_args()

def generate_lines_angle(matrix_shape, angle_deg):
    """
    严格沿指定角度生成矩阵中的线段坐标
    不生成水平或竖直的线段
    """
    h, w = matrix_shape
    theta = math.radians(angle_deg)
    dx = math.cos(theta)
    dy = math.sin(theta)

    if abs(dx) < 1e-6:  # 垂直线
        x_dir_only = True
    else:
        x_dir_only = False

    lines = []

    max_dim = int(np.ceil(np.hypot(h, w)))  # 最大长度

    if 0 < angle_deg < 90:
        # 从左边界
        for y0 in range(h):
            x0 = 0
            line = []
            for t in range(max_dim):
                x = int(round(x0 + dx * t))
                y = int(round(y0 + dy * t))
                if 0 <= x < w and 0 <= y < h:
                    line.append((y, x))
                else:
                    break
            if len(line) > 1:
                lines.append(line)

        # 从上边界
        for x0 in range(w):
            y0 = 0
            line = []
            for t in range(max_dim):
                x = int(round(x0 + dx * t))
                y = int(round(y0 + dy * t))
                if 0 <= x < w and 0 <= y < h:
                    line.append((y, x))
                else:
                    break
            if len(line) > 1:
                lines.append(line)

    elif 90 < angle_deg < 180:
        # 从上边界
        for x0 in range(w):
            y0 = 0
            line = []
            for t in range(max_dim):
                x = int(round(x0 + dx * t))
                y = int(round(y0 + dy * t))
                if 0 <= x < w and 0 <= y < h:
                    line.append((y, x))
                else:
                    break
            if len(line) > 1:
                lines.append(line)

        # 从右边界
        for y0 in range(h):
            x0 = w - 1
            line = []
            for t in range(max_dim):
                x = int(round(x0 + dx * t))
                y = int(round(y0 + dy * t))
                if 0 <= x < w and 0 <= y < h:
                    line.append((y, x))
                else:
                    break
            if len(line) > 1:
                lines.append(line)

    elif angle_deg == 0:
        # 水平线
        for y0 in range(h):
            line = [(y0, x) for x in range(w)]
            lines.append(line)
    elif angle_deg == 90:
        # 垂直线
        for x0 in range(w):
            line = [(y, x0) for y in range(h)]
            lines.append(line)

    return lines


def plot_lines_on_matrix(matrix, lines, save_path=None, show=True):
    """
    将线段轨迹画在矩阵上，每条线独立显示
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix, cmap='gray', origin='upper')

    for line in lines:
        if len(line) < 2:
            continue
        y_coords, x_coords = zip(*line)
        plt.plot(x_coords, y_coords, linewidth=0.5, alpha=0.5, color='red')  # 每条线单独plot

    plt.title("Line trajectories")
    plt.axis('off')  # 可选：隐藏坐标轴
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)
    if show:
        plt.show()
    plt.close()


if __name__ == "__main__":

    args = get_args()

    if Path(args.file_path).suffix == ".grd":
        matrix = grd_to_numpy(args.file_path)
    elif Path(args.file_path).suffix == ".npy":
        matrix = np.load(args.file_path)
    else:
        raise ValueError("不支持的文件格式")

    print("矩阵大小:", matrix.shape)

    # angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    angles = [0,90]

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
            lines = generate_lines_angle(matrix.shape, angle)
            plot_lines_only_random_color(matrix, lines, save_path=png_dir / f"lines_angle_{angle}.png", show=args.vis)
            # # 可视化线段轨迹
            # if args.show_lines:
            #     plot_lines_on_matrix(matrix, lines, save_path=png_dir / f"lines_angle_{angle}.png", show=args.vis)

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

    print("\n✔ 任意角度一维子域滤波完成并绘制轨迹")
