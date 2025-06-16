import argparse
import tqdm
from multiprocessing import Pool, cpu_count
from utils import *
from grid_clip import *

def process_submatrix(sub_matrix, clip_method):
    """处理单个子矩阵的函数，用于多进程并行"""
    index = sub_matrix[0]
    sub_matrix_data = sub_matrix[1]
    clip_grids = eval(clip_method+"_clip_method")(sub_matrix_data)
    min_mean, min_msd = min_mse_average(clip_grids)
    return index, min_mean

def get_args():
    parser = argparse.ArgumentParser(description='Subdomain filtering')
    parser.add_argument('--epoch', type=int, default=5, help='迭代次数')
    parser.add_argument('--file_path', type=str, default=r"D:\Code\small_domain_filtering\output\cube-size5-polygon56\20250616-2013-38\iter_3data.grd", help='重力异常文件地址,目前支持xlsx npy 文件')
    parser.add_argument('--clip_method', type=str, default="polygon56", help='子域划分类型，可选 mi/ tian/ hua/ polygon56 ')
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
        submatrices = get_submatrices(matrix, subdomain_size)
        assert len(submatrices) == matrix.shape[0] * matrix.shape[1]

        output_matrix = np.zeros_like(matrix)

        # 使用进程池并行处理子矩阵
        with Pool(processes=processes) as pool:
            # 创建偏函数，固定clip_method参数
            from functools import partial
            process_func = partial(process_submatrix, clip_method=clip_method)

            results = list(tqdm.tqdm(
                pool.imap(process_func, submatrices),
                total=len(submatrices),
                desc="处理子域"
            ))

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