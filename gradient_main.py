import argparse
from multiprocessing import cpu_count
from utils.utils import *
from utils.grid_clip import *
from scipy.ndimage import map_coordinates

def sample_along_tangent(matrix, tangent_direction, num_points=5, step=1.0):
    """
    在每个点的切线上采样值
    :param matrix: 原始矩阵 (2D)
    :param tangent_direction: 每个点的切线方向 (弧度)
    :param num_points: 采样点数，默认5 (中心点±2)
    :param step: 采样间隔，默认1个像素
    :return: samples (rows, cols, num_points)
    """
    rows, cols = matrix.shape
    half = num_points // 2
    samples = np.zeros((rows, cols, num_points), dtype=float)

    # 坐标网格
    y_coords, x_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")

    for k in range(-half, half+1):
        dx = k * step * np.cos(tangent_direction)
        dy = - k * step * np.sin(tangent_direction)

        # 新坐标
        sample_x = x_coords + dx
        sample_y = y_coords + dy

        # 插值 (注意 map_coordinates 输入是 (行=y, 列=x))
        values = map_coordinates(matrix, [sample_y.ravel(), sample_x.ravel()],
                                 order=1, mode='nearest').reshape(rows, cols)

        samples[:, :, k+half] = values

    return samples


def calculate_tangent_equations(matrix, tangent_direction):
    """
    计算每个点的切线方程
    :param matrix: 2D 数据矩阵
    :param tangent_direction: 每个点的切线方向 (弧度)
    :return: equations 列表，每个元素是 (x0, y0, slope, eq_str)
    """
    rows, cols = matrix.shape
    equations = []

    for i in range(rows):
        for j in range(cols):
            x0, y0 = j, i   # 注意：matrix[y, x]，所以 j 是 x，i 是 y
            theta = tangent_direction[i, j]

            # 判断是否竖直
            if np.isclose(np.cos(theta), 0, atol=1e-6):
                eq_str = f"x = {x0}"
                slope = None
            else:
                slope = np.tan(theta)
                eq_str = f"y - {y0} = {slope:.4f} * (x - {x0})"

            equations.append((x0, y0, slope, eq_str))

    return equations



def calculate_tangent_direction(matrix):
    """计算矩阵每个点的切线方向（最大斜率方向的切线）和斜率大小"""
    # 计算x和y方向的一阶偏导数
    dx, dy = np.gradient(matrix)

    # 梯度大小（斜率强度）
    grad_magnitude = np.sqrt(dx**2 + dy**2)

    # 梯度方向（法向量方向）
    grad_direction = np.arctan2(dy, dx)

    # 切线方向 = 梯度方向 + 90度
    tangent_direction = grad_direction  + np.pi / 2

    return tangent_direction, grad_magnitude

def plot_tangent(matrix, directions, magnitudes, save_path, show_plot=False):
    """可视化切线方向和斜率大小"""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='gray', alpha=0.5)

    # 每隔一定间隔绘制箭头
    step = max(1, int(min(matrix.shape) / 50))
    x = np.arange(0, matrix.shape[1], step)
    y = np.arange(0, matrix.shape[0], step)
    X, Y = np.meshgrid(x, y)

    # 计算箭头的x和y分量
    U = np.cos(directions[::step, ::step])
    V = np.sin(directions[::step, ::step])

    # 根据斜率大小设置箭头颜色
    grad_mag = magnitudes[::step, ::step]
    norm = cm.colors.Normalize(vmin=np.min(grad_mag), vmax=np.max(grad_mag))

    plt.quiver(X, Y, U, V, grad_mag, cmap='jet', norm=norm,
               scale=50, width=0.002, headwidth=3)

    plt.colorbar(label='Slope Magnitude')
    plt.title('Tangent Directions (Max Slope)')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    plt.close()

def split_odd_list(lst):
    """
    将奇数长度的列表分为三份，其中第二份与第一份尺寸相同

    参数:
        lst: 奇数长度的列表

    返回:
        包含三个子列表的元组，分别为：
        - 第一份：从开始到中间元素（包含中间元素）
        - 第二份：以中间元素为中心，长度与第一份相同的子列表
        - 第三份：从中间元素到结尾（包含中间元素）

    异常:
        如果输入列表长度为偶数，将抛出ValueError
    """
    # 检查列表长度是否为奇数
    if len(lst) % 2 == 0:
        raise ValueError("输入列表必须是奇数长度")

    # 计算中间元素的索引
    mid_index = len(lst) // 2

    # 第一份：从开始到中间元素（包含）
    part1 = lst[:mid_index + 1]
    part1_length = len(part1)

    # 第二份：以中间元素为中心，长度与第一份相同
    # 计算第二份的起始索引
    start_index = mid_index - (part1_length//2)
    # 确保起始索引不为负数（对于长度为1的列表）
    start_index = max(0, start_index)
    part2 = lst[start_index:start_index + part1_length]

    # 第三份：从中间元素到最后（包含）
    part3 = lst[mid_index:]

    return part1, part2, part3



def get_args():
    parser = argparse.ArgumentParser(description='Subdomain filtering with tangent analysis')
    parser.add_argument('--epoch', type=int, default=5, help='迭代次数')
    parser.add_argument('--file_path', type=str, default=r"D:\Code\small_domain_filtering\data\small.xlsx", help='重力异常文件地址,目前支持xlsx npy 文件')
    parser.add_argument('--subdomain_size', type=int, default=5, help="子域大小,只能为奇数")
    parser.add_argument('--output', type=str, default="output", help='保存路径')
    parser.add_argument('--vis', type=bool, default=False, help='是否可视化等高线图')
    parser.add_argument('--plot_levels', type=int, default=50, help='绘制等高线的levels')
    parser.add_argument('--plot_type', type=str, default="filled", help='绘制等高线的类型，可选 filled/ contour/ 3d')
    parser.add_argument('--processes', type=int, default=None, help='并行进程数，默认使用CPU核心数')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    epoch = args.epoch
    file_path = args.file_path

    subdomain_size = args.subdomain_size
    output = args.output
    vis = args.vis
    plot_levels = args.plot_levels
    plot_type = args.plot_type
    processes = args.processes if args.processes else cpu_count() - 1

    time = get_current_date_formatted()
    stem = Path(file_path).stem
    output_path = os.path.join(output, stem+"-"+ "size" + str(subdomain_size)+"-"+"gradient", time)
    output_path_png = os.path.join(output_path,"png")
    output_path_grd = os.path.join(output_path,"grd")
    output_path_xlsx = os.path.join(output_path,"xlsx")
    mkdir_if_not_exist(output_path_png)
    mkdir_if_not_exist(output_path_grd)
    mkdir_if_not_exist(output_path_xlsx)

    # 读取输入文件
    if Path(file_path).suffix == ".xlsx":
        matrix = excel_to_numpy(file_path)
    elif Path(file_path).suffix == ".npy":
        matrix = np.load(file_path)
    elif Path(file_path).suffix == ".grd":
        matrix = grd_to_numpy(file_path)
    else:
        raise ValueError("暂时不支持其他格式文件")
    print("矩阵大小:", matrix.shape)



    # 绘制原始数据等高线
    plot_contour(matrix,
                 levels=plot_levels,
                 title="raw_data",
                 plot_type=plot_type,
                 save_path=os.path.join(output_path_png,"raw_data.png"),
                 show_plot=vis)

    # 计算切线方向
    print("计算切线方向...")
    tangent_direction, slope_magnitude = calculate_tangent_direction(matrix)
    plot_tangent(
        matrix,
        tangent_direction,
        slope_magnitude,
        save_path=os.path.join(output_path_png, "raw_data_tangent.png"),
        show_plot=vis
    )

    for k in range(epoch):
        print("--------")
        print(f"- 正在进行第 {k+1} 轮迭代...")

        output_matrix = np.zeros_like(matrix)
        tangent_direction, slope_magnitude = calculate_tangent_direction(matrix)
        samples = sample_along_tangent(matrix, tangent_direction, num_points=subdomain_size, step=1.0)
        # np.save(os.path.join(output_path_grd, "tangent_samples.npy"), samples)
        for i in range(samples.shape[0]):
            for j in range(samples.shape[1]):
                win = samples[i, j, :]
                window_size = len(win)
                median = window_size // 2
                odd_list = split_odd_list(win) # 将列表三分
                min_mean, min_msd = min_mse_average(odd_list)
                output_matrix[i, j] = min_mean

        plot_contour(output_matrix,
                     levels=plot_levels,
                     title="iter_"+str(i+1)+"data",
                     plot_type=plot_type,
                     save_path=os.path.join(output_path_png,"iter_"+str(k+1)+"data.png"),
                     show_plot=vis)
        matrix = output_matrix

    print("处理完成，结果已保存至:", output_path)
