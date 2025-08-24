import argparse

import tqdm
from multiprocessing import cpu_count
from utils.utils import *
from utils.grid_clip import *


def partial_derivative_x(x, y, G, M, D):
    """计算 g 对 x 的偏导数"""
    denominator = (x**2 + y**2 + D**2) ** 2.5
    return -(3 * G * M * D * x) / denominator
def partial_derivative_y(x, y, G, M, D):
    """计算 g 对 y 的偏导数"""
    denominator = (x**2 + y**2 + D**2) ** 2.5
    return -(3 * G * M * D * y) / denominator

def min3_mse(clip_grids):
    mean_list = []
    msd_list = []
    for grids in clip_grids:
        grids = np.array(grids)
        mean = np.mean(grids)
        msd = np.mean((grids - mean) ** 2)
        mean_list.append(mean)
        msd_list.append(msd)

    # 获取最小的三个均方差的索引
    min_indices = np.argsort(msd_list)[:3]

    # 根据索引获取对应的均值和均方差
    result = []
    for idx in min_indices:
        result.append(clip_grids[idx])

    return min_indices, result

def min_mse(clip_grids):
    mean_list = []
    msd_list = []
    for grids in clip_grids:
        grids = np.array(grids)
        mean = np.mean(grids)
        msd = np.mean((grids - mean) ** 2)
        mean_list.append(mean)
        msd_list.append(msd)

    min_index = np.argmin(msd_list)
    return min_index

def process_submatrix(sub_matrix, sub_pd_matrix, clip_method):
    """处理单个子矩阵的函数，用于多进程并行"""
    index = sub_matrix[0]
    sub_matrix_data = sub_matrix[1]
    clip_grids = eval(clip_method+"_clip_method")(sub_matrix_data)
    pd_clip_grids = eval(clip_method+"_clip_method")(sub_pd_matrix[1])

    min3_index, min3_clip_grids = min3_mse(clip_grids)

    min3_pd_clip_grids = []
    for idx in min3_index:
        min3_pd_clip_grids.append(pd_clip_grids[idx])

    min_index = min_mse(min3_pd_clip_grids)
    min_mean = np.mean(min3_clip_grids[min_index])

    return index, min_mean

def calculate_horizontal_derivative(matrix):
    # 只针对球体
    pd_matrix = np.zeros_like(matrix)
    G = 6.67*1e-3
    R = 40 # 球半径
    D = 50 # 深度
    sigma = 1 # 密度
    M=(4/3)*np.pi*(R*R*R)*sigma

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            x = j*1 - 80 # 球中心的位置
            y = i*1 - 80 # 球中心的位置
            pd_x = partial_derivative_x(x, y, G, M, D)
            pd_y = partial_derivative_y(x, y, G, M, D)
            pd_matrix[i][j] = np.sqrt(pd_x**2 + pd_y**2)
    return pd_matrix

def get_args():
    parser = argparse.ArgumentParser(description='Subdomain filtering')
    parser.add_argument('--epoch', type=int, default=5, help='迭代次数')
    parser.add_argument('--file_path', type=str, default=r"D:\Code\small_domain_filtering\data\sphere\sphere.xlsx", help='重力异常文件地址,目前支持xlsx npy 文件')
    parser.add_argument('--clip_method', type=str, default="hua", help='子域划分类型，可选 mi/ tian/ hua/ polygon56 ')
    parser.add_argument('--subdomain_size', type=int, default=9, help="子域大小,只能为奇数")
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
    clip_method = args.clip_method
    subdomain_size = args.subdomain_size
    output = args.output
    vis = args.vis
    plot_levels = args.plot_levels
    plot_type = args.plot_type
    processes = args.processes if args.processes else cpu_count() - 1

    time = get_current_date_formatted()
    stem = Path(file_path).stem
    output_path = os.path.join(output, stem+"-"+ "size" + str(subdomain_size)+"-"+clip_method, time)
    output_path_png = os.path.join(output_path,"png")
    output_path_grd = os.path.join(output_path,"grd")
    output_path_xlsx = os.path.join(output_path,"xlsx")
    mkdir_if_not_exist(output_path_png)
    mkdir_if_not_exist(output_path_grd)
    mkdir_if_not_exist(output_path_xlsx)

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
                 save_path=os.path.join(output_path_png,"raw_data.png"),
                 show_plot=vis)

    for i in range(epoch):
        print("--------")
        print(f"- 正在进行第 {i+1} 轮迭代...")

        pd_matrix = calculate_horizontal_derivative(matrix)

        submatrices = get_submatrices(matrix, subdomain_size)
        sub_pd_matrixs = get_submatrices(pd_matrix, subdomain_size)

        assert len(submatrices) == matrix.shape[0] * matrix.shape[1]
        output_matrix = np.zeros_like(matrix)

        results = []
        for k in tqdm.tqdm(range(len(submatrices))):
            results.append(process_submatrix(submatrices[k],sub_pd_matrixs[k],clip_method))

        # 更新结果矩阵
        for index, value in results:
            output_matrix[index[0], index[1]] = value

        matrix = output_matrix

        plot_contour(output_matrix,
                     levels=plot_levels,
                     title="iter_"+str(i+1)+"data",
                     plot_type=plot_type,
                     save_path=os.path.join(output_path_png,"iter_"+str(i+1)+"data.png"),
                     show_plot=vis)
        save_grd(output_matrix,os.path.join(output_path_grd,"iter_"+str(i+1)+"data.grd"))
        save_xlsx(output_matrix,os.path.join(output_path_xlsx,"iter_"+str(i+1)+"data.xlsx"))
        print(f"- 第 {i+1} 轮迭代结果已保存在{output_path}")

    print("")
    print("- End, Wishing you a wonderful day! ")