import argparse
import tqdm
from utils import *
from grid_clip import *

def get_args():
    parser = argparse.ArgumentParser(description='Subdomain filtering')
    parser.add_argument('--epoch', type=int, default=5, help='迭代次数')
    parser.add_argument('--file_path', type=str, default="data/cube/cube.xlsx", help='重力异常文件地址,目前支持xlsx npy 文件')
    parser.add_argument('--clip_method', type=str, default="mi", help='子域划分类型，可选 mi/ tian ')
    parser.add_argument('--subdomain_size', type=int, default=9, help="子域大小,只能为奇数")
    parser.add_argument('--output', type=str, default="output", help='保存路径')
    parser.add_argument('--vis', type=bool, default=False, help='是否可视化等高线图')
    parser.add_argument('--plot_levels', type=int, default=50, help='绘制等高线的levels')
    parser.add_argument('--plot_type', type=str, default="filled", help='绘制等高线的类型，可选 filled/ contour/ 3d')
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

    time = get_current_date_formatted()
    stem = Path(file_path).stem
    output_path = os.path.join(output, stem+"-"+ "size" + str(subdomain_size)+"-"+clip_method, time)
    mkdir_if_not_exist(output_path)

    if Path(file_path).suffix == ".xlsx":
        matrix = excel_to_numpy(file_path)
    elif Path(file_path).suffix == ".npy":
        matrix = np.load(file_path)
    else:
        ValueError("暂时不支持其他格式文件")
    print("矩阵大小:", matrix.shape)

    plot_contour(matrix,
                 levels=plot_levels,
                 title="raw_data",
                 plot_type=plot_type,
                 save_path=os.path.join(output_path,"raw_data.png"),
                 show_plot=vis)

    for i in range(epoch):
        print("--------")
        print(f"- 正在进行第 {i+1} 轮迭代...")
        submatrices = get_submatrices(matrix, subdomain_size)
        assert len(submatrices) == matrix.shape[0] * matrix.shape[1]
        # print("获得子矩阵个数:", len(submatrices))
        output_matrix = np.zeros_like(matrix)
        for sub_matrix in tqdm.tqdm(submatrices):
            index = sub_matrix[0]
            sub_matrix = sub_matrix[1]
            clip_grids = eval(clip_method+"_clip_method")(sub_matrix)
            min_mean, min_msd = min_mse_average(clip_grids)
            output_matrix[index[0],index[1]] = min_mean
        matrix = output_matrix

        plot_contour(output_matrix,
                     levels=plot_levels,
                     title="iter_"+str(i+1)+"data",
                     plot_type=plot_type,
                     save_path=os.path.join(output_path,"iter_"+str(i+1)+"data.png"),
                     show_plot=vis)
        print(f"- 第 {i+1} 轮迭代结果已保存在{output_path}")


print("")
print("- End, Wishing you a wonderful day! ")