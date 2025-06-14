import openpyxl
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, List
from datetime import datetime
import os
def excel_to_numpy(
        file_path: Union[str, Path],
        sheet_name: str = None,
        dtype: np.dtype = np.float64
) -> np.ndarray:

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    if file_path.suffix.lower() != '.xlsx':
        raise ValueError(f"文件不是 .xlsx 格式: {file_path}")
    try:
        workbook = openpyxl.load_workbook(file_path, read_only=True)
        if sheet_name:
            if sheet_name not in workbook.sheetnames:
                raise ValueError(f"工作表 '{sheet_name}' 不存在")
            sheet = workbook[sheet_name]
        else:
            sheet = workbook.active
        data = []
        first_row = True
        for row in sheet.iter_rows(values_only=True):
            if first_row:
                first_row = False
                continue  # 跳过第一行
            # 跳过第一列并将剩余值转换为列表
            data.append(list(row[1:]))
        # 关闭工作簿
        workbook.close()
        # 转换为 NumPy 数组
        if not data:
            raise ValueError("Excel 文件中没有数据")
        array = np.array(data, dtype=dtype)
        # print(f"成功转换为 {array.shape} 的 NumPy 矩阵")
        return array
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return np.array([])

def grd_to_numpy(file_path):
    from osgeo import gdal
    dataset = gdal.Open(file_path)
    if not dataset:
        raise RuntimeError("无法打开 GRD 文件")

    # 读取第一个波段
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    return data



def plot_contour(
        matrix: np.ndarray,
        title: str = "data",
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = "viridis",
        levels: Optional[Union[int, List[float]]] = None,
        show_colorbar: bool = False,
        plot_type: str = "filled",
        save_path: Optional[str] = None,
        show_plot: bool = True
) -> None:
    """
    将 NumPy 二维矩阵绘制成等高线图

    参数:
        matrix: 输入的二维 NumPy 矩阵
        title: 图表标题
        figsize: 图表大小
        cmap: 颜色映射名称
        levels: 等高线级别数或自定义级别列表
        show_colorbar: 是否显示颜色条
        plot_type: 图表类型，可选 'filled'（填充等高线）、'contour'（基本等高线）或 '3d'（3D 表面图）
        save_path: 图表保存路径，若为 None 则不保存
        show_plot: 是否显示图表
    """
    # 创建网格坐标
    x = np.arange(matrix.shape[1])
    y = np.arange(matrix.shape[0])
    X, Y = np.meshgrid(x, y)

    # 创建图形
    plt.figure(figsize=figsize)

    if plot_type == 'filled':
        # 填充等高线图
        contour = plt.contourf(X, Y, matrix, levels=levels, cmap=cmap)
        plt.contour(X, Y, matrix, levels=contour.levels, colors='k', linewidths=0.5)
        plt.title(f"{title} (filled)")

    elif plot_type == 'contour':
        # 基本等高线图
        contour = plt.contour(X, Y, matrix, levels=levels, cmap=cmap, linewidths=2)
        plt.clabel(contour, inline=True, fontsize=8)
        plt.title(f"{title} (contour)")

    elif plot_type == '3d':
        # 3D 表面图
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, matrix, cmap=cmap, linewidth=0, antialiased=True)
        ax.set_title(f"{title} (3D)")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if show_colorbar:
            fig.colorbar(surf, shrink=0.5, aspect=5)
    else:
        raise ValueError(f"不支持的图表类型: {plot_type}，请选择 'filled', 'contour' 或 '3d'")

    # 设置坐标轴标签和颜色条
    if plot_type != '3d':
        plt.xlabel('X')
        plt.ylabel('Y')
        if show_colorbar:
            plt.colorbar(label='nums')

    plt.tight_layout()

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # print(f"图表已保存至: {save_path}")

    # 显示图表
    if show_plot:
        plt.show()

def get_submatrices(matrix, n):
    """
    遍历一个二维 numpy 矩阵，以每个元素为中心提取边长为 n 的子矩阵。
    :param matrix: 输入的二维 numpy 数组
    :param n: 子矩阵的边长，必须为奇数
    :return: 包含所有可提取子矩阵的列表，格式为 ((i, j), submatrix)
    """
    assert n % 2 == 1, "n 必须是奇数"
    pad = n // 2
    # padded_matrix = np.pad(matrix, pad_width=pad, mode='constant', constant_values=0)
    padded_matrix = np.pad(matrix, pad_width=pad, mode='edge')
    submatrices = []
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            submatrix = padded_matrix[i:i+n, j:j+n]
            submatrices.append(((i, j), submatrix))
    return submatrices

def min_mse_average(clip_grids):
    mean_list = []
    msd_list = []
    for grids in clip_grids:
        grids = np.array(grids)
        mean = np.mean(grids)
        msd = np.mean((grids - mean) ** 2)
        mean_list.append(mean)
        msd_list.append(msd)

    min_index = np.argmin(msd_list)
    min_mean = mean_list[min_index]
    min_msd = msd_list[min_index]
    return min_mean, min_msd

def get_current_date_formatted():
    # 获取当前时间
    now = datetime.now()
    # 格式化为 YYYYMMDD 形式
    formatted_date = now.strftime('%Y%m%d-%H%M-%S')
    return formatted_date

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)