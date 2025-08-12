import argparse
import tqdm
from utils import *
from grid_clip import *

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
    parser.add_argument('--file_path', type=str, default=r"E:\code\my_project\small_domain_filtering\data\two_cube_output.xlsx", help='重力异常文件地址,目前支持xlsx文件')
    parser.add_argument('--subdomain_size', type=int, default=5, help="子域大小,只能为奇数")
    parser.add_argument('--output', type=str, default="output", help='保存路径')
    parser.add_argument('--plot_levels', type=int, default=50, help='绘制等高线的levels')
    parser.add_argument('--plot_type', type=str, default="filled", help='绘制等高线的类型，可选 filled/ contour/ 3d')
    parser.add_argument('--vis', type=bool, default=False, help='是否可视化等高线图')
    # parser.add_argument('--processes', type=int, default=None, help='并行进程数，默认使用CPU核心数')
    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = get_args()
    epoch = args.epoch
    file_path = args.file_path
    subdomain_size = args.subdomain_size
    plot_levels = args.plot_levels
    plot_type = args.plot_type
    output = args.output
    vis = args.vis

    time = get_current_date_formatted()
    stem = Path(file_path).stem
    output_path = os.path.join(output, stem + "-" + "size_5" , time)
    output_path_png = os.path.join(output_path, "png")
    mkdir_if_not_exist(output_path_png)

    if Path(file_path).suffix == ".xlsx":
        matrix = excel_to_numpy(file_path)
    elif Path(file_path).suffix == ".npy":
        matrix = np.load(file_path)
    elif Path(file_path).suffix == ".grd":
        matrix = grd_to_numpy(file_path)
    else:
        raise ValueError("暂时不支持其他格式文件")
    print("矩阵大小:", matrix.shape)

    plot_contour(matrix,
                 levels=plot_levels,
                 title="raw_data",
                 plot_type=plot_type,
                 save_path=os.path.join(output_path_png, "raw_data.png"),
                 show_plot=vis)

    for i in range(epoch):
        print("--------")
        print(f"- 正在进行第 {i + 1} 轮迭代...")
        # 行二维计算
        row_result = []
        for row in range(matrix.shape[0]):
            data = matrix[row, :].tolist()
            data = centered_moving_window_variance(data, subdomain_size)
            row_result.append(data)
        row_result = np.array(row_result)
        plot_contour(row_result,
                     levels=plot_levels,
                     title="iter_" + str(i + 1) + "data",
                     plot_type=plot_type,
                     save_path=os.path.join(output_path_png, "iter_" + str(i + 1) + "_row.png"),
                     show_plot=vis)
        # 列二维计算
        col_result = []
        for col in range(matrix.shape[1]):
            data = matrix[:, col].tolist()
            data = centered_moving_window_variance(data, subdomain_size)
            col_result.append(data)
        col_result = np.array(col_result).T

        plot_contour(col_result,
                     levels=plot_levels,
                     title="iter_" + str(i + 1) + "data",
                     plot_type=plot_type,
                     save_path=os.path.join(output_path_png, "iter_" + str(i + 1) + "_col.png"),
                     show_plot=vis)

        result = row_result + col_result
        matrix = result
        plot_contour(result,
                     levels=plot_levels,
                     title="iter_" + str(i + 1) + "data",
                     plot_type=plot_type,
                     save_path=os.path.join(output_path_png, "iter_" + str(i + 1) + "_data.png"),
                     show_plot=vis)
        print(f"- 第 {i + 1} 轮迭代结果已保存在{output_path}")

    print("")
    print("- End, Wishing you a wonderful day! ")
