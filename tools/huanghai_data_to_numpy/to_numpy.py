"""
将黄海重力异常 Excel（经度、纬度、重力异常三列）转为二维 numpy 网格并可视化。
前两列为经纬度坐标，第三列为重力异常值。
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import openpyxl

# 项目根目录加入路径，便于引用 utils
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.utils import plot_matrix_huanghai, plot_matrix_huanghai_interactive

# 默认路径
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_XLSX = SCRIPT_DIR / "zhujiang.xlsx"
DEFAULT_NPY = SCRIPT_DIR / "zhujiang_gravity.npy"
DEFAULT_FIG = SCRIPT_DIR / "zhujiang_gravity.png"


def load_xyz_from_excel(xlsx_path: Path):
    """读取 xlsx：第 1 列经度，第 2 列纬度，第 3 列重力异常。返回 (lon, lat, value) 数组。"""
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"文件不存在: {xlsx_path}")
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    if not rows:
        raise ValueError("Excel 文件中没有数据")
    # 第一行可能是表头，若首单元格可转为浮点则当作数据
    data_rows = []
    for row in rows:
        if len(row) < 3:
            continue
        try:
            a, b, c = float(row[0]), float(row[1]), float(row[2])
            if np.isfinite(a) and np.isfinite(b) and np.isfinite(c):
                data_rows.append([a, b, c])
        except (TypeError, ValueError):
            continue
    if not data_rows:
        raise ValueError("未找到有效的三列数值（经度、纬度、重力异常）")
    arr = np.array(data_rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def xyz_to_grid(lon, lat, values, nrows=None, ncols=None):
    """
    将散点 (lon, lat, values) 转为规则二维网格。
    若经纬度本身构成规则网格则按网格填充，否则用线性插值到规则网格。
    """
    from scipy.interpolate import griddata

    lon_min, lon_max = lon.min(), lon.max()
    lat_min, lat_max = lat.min(), lat.max()

    # 若未指定网格大小，根据唯一经纬度数量或点数估计
    n_pts = len(lon)
    n_lon_unique = len(np.unique(np.round(lon, 6)))
    n_lat_unique = len(np.unique(np.round(lat, 6)))
    if nrows is None:
        nrows = n_lat_unique if n_lat_unique > 1 else int(np.sqrt(n_pts))
    if ncols is None:
        ncols = n_lon_unique if n_lon_unique > 1 else int(np.sqrt(n_pts))

    # 规则网格
    xi = np.linspace(lon_min, lon_max, ncols)
    yi = np.linspace(lat_max, lat_min, nrows)  # 纬度从高到低，使行与北对应
    X, Y = np.meshgrid(xi, yi)
    points = np.column_stack((lon, lat))
    grid_values = griddata(points, values, (X, Y), method="linear", fill_value=np.nan)
    # 外推：用最近邻填边缘 NaN（可选）
    if np.any(np.isnan(grid_values)):
        grid_fill = griddata(points, values, (X, Y), method="nearest", fill_value=np.nan)
        nan_mask = np.isnan(grid_values)
        grid_values[nan_mask] = grid_fill[nan_mask]
    return grid_values, xi, yi


def main():
    import argparse
    parser = argparse.ArgumentParser(description="黄海重力异常 Excel 转二维 numpy 并可视化")
    parser.add_argument("--input", "-i", type=str, default=str(DEFAULT_XLSX), help="输入 xlsx 路径")
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_NPY), help="输出 npy 路径")
    parser.add_argument("--fig", "-f", type=str, default=str(DEFAULT_FIG), help="输出图像路径")
    parser.add_argument("--rows", type=int, default=None, help="输出网格行数（默认自动）")
    parser.add_argument("--cols", type=int, default=None, help="输出网格列数（默认自动）")
    parser.add_argument("--no-show", action="store_true", help="不弹出显示窗口，只保存文件")
    parser.add_argument("--interactive", action="store_true", help="打开可手动调节色条（vmin/vmax/黄色起始）的交互窗口")
    args = parser.parse_args()

    xlsx_path = Path(args.input)
    out_npy = Path(args.output)
    out_fig = Path(args.fig)

    print(f"读取: {xlsx_path}")
    lon, lat, gravity = load_xyz_from_excel(xlsx_path)
    print(f"  点数: {len(lon)}, 经度范围: [{lon.min():.4f}, {lon.max():.4f}], 纬度范围: [{lat.min():.4f}, {lat.max():.4f}]")

    grid, xi, yi = xyz_to_grid(lon, lat, gravity, nrows=args.rows, ncols=args.cols)
    print(f"  网格形状: {grid.shape} (行×列 = 纬度×经度)")

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, grid)
    print(f"已保存 numpy: {out_npy}")

    # 可视化
    extent = (float(xi[0]), float(xi[-1]), float(yi[-1]), float(yi[0]))
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    if args.interactive:
        plot_matrix_huanghai_interactive(
            grid,
            extent=extent,
            colorbar_label="重力异常 / (10⁻⁵ m/s²)",
            initial_vmin=-30.0,
            initial_vmax=45.0,
            initial_yellow_at=15.0,
            save_path=str(out_fig),
        )
        print("交互窗口已关闭。若在窗口中点击过「保存当前视图」，图像已保存。")
    else:
        plot_matrix_huanghai(
            grid,
            extent=extent,
            save_path=str(out_fig),
            show_plot=not args.no_show,
            show_colorbar=True,
            colorbar_label="重力异常 / (10⁻⁵ m/s²)",
            colorbar_interval=5.0,
            vmin_display=-50.0,
            vmax_display=60.0,
            use_figure_cbar=True,
        )
        print(f"已保存图像: {out_fig}")


if __name__ == "__main__":
    main()
