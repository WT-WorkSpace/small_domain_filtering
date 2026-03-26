import openpyxl
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, List, Any
from datetime import datetime
import os
from openpyxl import Workbook

import matplotlib as mpl




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


def save_grd(array, output_path, x_size=None, y_size=None,
                 x_min=None, y_min=None, pixel_width=None, pixel_height=None):
    """
    将numpy数组保存为GRD文件

    参数:
        array: numpy数组，要保存的栅格数据
        output_path: str, 输出GRD文件的路径
        x_size, y_size: 栅格的列数和行数，如果未提供则使用array的形状
        x_min, y_min: 栅格左上角的坐标，如果未提供则默认为0
        pixel_width, pixel_height: 像元宽度和高度，如果未提供则默认为1

    返回:
        成功保存返回True，否则抛出异常
    """
    from osgeo import gdal, osr

    # 获取数组的尺寸
    if y_size is None and x_size is None:
        y_size, x_size = array.shape
    elif y_size is None:
        y_size = array.shape[0]
    elif x_size is None:
        x_size = array.shape[1]

    # 设置默认地理参考参数
    if x_min is None:
        x_min = 0
    if y_min is None:
        y_min = 0
    if pixel_width is None:
        pixel_width = 1
    if pixel_height is None:
        pixel_height = 1

    # 创建输出驱动
    driver = gdal.GetDriverByName("GTiff")
    # 创建输出数据集
    dataset = driver.Create(
        output_path, x_size, y_size, 1, gdal.GDT_Float32)

    if dataset is None:
        raise RuntimeError(f"无法创建GRD文件: {output_path}")

    geotransform = (x_min, pixel_width, 0, y_min, 0, -pixel_height)
    dataset.SetGeoTransform(geotransform)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # WGS84坐标系
    dataset.SetProjection(srs.ExportToWkt())

    band = dataset.GetRasterBand(1)
    band.WriteArray(array)

    # 刷新缓存，确保数据写入文件
    band.FlushCache()

    # 关闭数据集
    dataset = None
    return True

def save_xlsx(array, output_path):

    wb = Workbook()
    ws = wb.active

    # 获取数组的维度
    rows, columns = array.shape

    # 将数组数据逐行写入 Excel
    for i in range(rows):
        for j in range(columns):
            # 注意：Excel 行和列索引从 1 开始
            ws.cell(row=i+1, column=j+1, value=array[i, j])

    # 保存工作簿
    wb.save(filename=output_path)
    return True

def numpy_to_xlsx(array, output_path, headers=None, sheet_name="Sheet1"):

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name

    # 添加表头（如果提供）
    if headers is not None:
        for j, header in enumerate(headers):
            ws.cell(row=1, column=j+1, value=header)

    # 确定数据起始行
    start_row = 2 if headers is not None else 1

    # 获取数组的维度
    rows, columns = array.shape

    # 写入数据
    for i in range(rows):
        for j in range(columns):
            ws.cell(row=i+start_row, column=j+1, value=array[i, j])

    wb.save(filename=output_path)
    return True



def plot_contour(
        matrix: np.ndarray,
        title: str = "data",
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = "viridis",
        levels: Optional[Union[int, List[float]]] = None,
        show_colorbar: bool = False,
        plot_type: str = "filled",
        save_path: Optional[str] = None,
        show_plot: bool = True,
        y_origin: str = "upper",      # 新增：'upper' 表示第0行在上方（图像坐标），'lower' 表示数学坐标
        aspect: Optional[str] = "equal"  # 新增：'equal'/'auto'/None
) -> None:
    """
    将 NumPy 二维矩阵绘制成等高线图

    红色=最大值，紫色=最小值（默认会强制使用类似 turbo 的配色，即使 cmap 仍保持入参不变）
    """
    if matrix.ndim != 2:
        raise ValueError("matrix 必须是二维数组")

    # ---- 颜色映射：保证 紫(小) -> 红(大) ----
    # 不改动入参，但默认入参是 viridis（不符合你想要的红最大/紫最小），因此这里内部做映射：
    # 1) 若用户没刻意指定（仍是默认 viridis），则改用 turbo
    # 2) 若 matplotlib 无 turbo，则用自定义紫->红渐变
    # cmap_to_use = cmap
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    purple_min = (150 / 255.0, 100 / 255.0, 255 / 255.0)
    if isinstance(cmap, str) and cmap == "viridis":
        cm = mpl.colors.LinearSegmentedColormap.from_list(
            "purple150_100_255_to_red",
            [
                purple_min,  # 最小值：指定紫色
                "#0033ff",  # 蓝
                "#00c8ff",  # 青
                "#00ff6a",  # 绿
                "#ffe600",  # 黄
                "#ff2a00",  # 红（最大值）
            ],
            N=256
        )
    else:
        try:
            cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        except Exception:
            # 万一用户给了奇怪的 cmap，退回到自定义
            cm = mpl.colors.LinearSegmentedColormap.from_list(
                "purple150_100_255_to_red_fallback",
                [purple_min, "#0033ff", "#00c8ff", "#00ff6a", "#ffe600", "#ff2a00"],
                N=256
            )

    # ---- 数值范围：保证颜色与最小/最大值对应稳定 ----
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # levels：如果没传，给一个更接近示例图的平滑层级数
    if levels is None:
        levels_to_use = 20
    else:
        levels_to_use = levels

    # 坐标网格（X 对应列索引，Y 对应行索引）
    ny, nx = matrix.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    if plot_type == '3d':
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(X, Y, matrix, cmap=cm, norm=norm, linewidth=0, antialiased=True)
        # ax.set_title(f"{title} (3D)")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if show_colorbar:
            fig.colorbar(surf, shrink=0.5, aspect=5)

        if y_origin == "upper":
            ax.set_ylim(ax.get_ylim()[::-1])

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close(fig)
        return

    # ---- 2D 情况 ----
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == 'filled':
        csf = ax.contourf(X, Y, matrix, levels=levels_to_use, cmap=cm, norm=norm)
        cs = ax.contour(X, Y, matrix, levels=csf.levels, colors='k', linewidths=1.5)
        # 加数字标注（示例图那样）
        # ax.clabel(cs, inline=True, fontsize=8, fmt='%g')

        # ax.set_title(f"{title} (filled)")
        if show_colorbar:
            fig.colorbar(csf, ax=ax, label='nums')

    elif plot_type == 'contour':
        cs = ax.contour(X, Y, matrix, levels=levels_to_use, cmap=cm, norm=norm, linewidths=1.5)
        # ax.clabel(cs, inline=True, fontsize=8, fmt='%g')
        # ax.set_title(f"{title} (contour)")
        if show_colorbar:
            fig.colorbar(cs, ax=ax, label='nums')

    else:
        raise ValueError(f"不支持的图表类型: {plot_type}，请选择 'filled', 'contour' 或 '3d'")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # 关键：第0行在上方（图像坐标）
    if y_origin == "upper":
        ax.invert_yaxis()

    if aspect:
        ax.set_aspect(aspect, adjustable='box')

    # x 轴刻度放到上方（用 ax，别用 plt.gca()）
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close(fig)


class _YellowBreakNorm(mpl.colors.Normalize):
    """将 [vmin, yellow_at] 映射到色标 0~yellow_pos（蓝→绿），[yellow_at, vmax] 映射到 yellow_pos~1（黄→红）"""
    def __init__(self, vmin, vmax, yellow_at, yellow_pos=6.0/21.0):
        self.yellow_at = yellow_at
        self.yellow_pos = float(yellow_pos)
        super().__init__(vmin, vmax)

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip
        result = np.ma.filled(np.ma.asarray(value), np.nan).astype(float)
        vmin, vmax = self.vmin, self.vmax
        ya, yp = self.yellow_at, self.yellow_pos
        if ya <= vmin:
            # 全部用黄~红段
            t = (result - vmin) / (vmax - vmin) * (1 - yp) + yp
        elif ya >= vmax:
            # 全部用蓝~黄段
            t = (result - vmin) / (vmax - vmin) * yp
        else:
            t = np.where(
                result <= ya,
                (result - vmin) / (ya - vmin) * yp,
                yp + (result - ya) / (vmax - ya) * (1 - yp)
            )
        if clip:
            t = np.clip(t, 0, 1)
        return np.ma.filled(np.ma.masked_invalid(t), 0)


class _BlueYellowBreakNorm(mpl.colors.Normalize):
    """[vmin, light_blue_at]→深蓝到亮蓝, [light_blue_at, yellow_at]→亮蓝经绿到黄, [yellow_at, vmax]→黄到红"""
    def __init__(self, vmin, vmax, light_blue_at=5.0, yellow_at=30.0,
                 light_blue_pos=2.0/23.0, yellow_pos=7.0/23.0):
        self.light_blue_at = float(light_blue_at)
        self.yellow_at = float(yellow_at)
        self.light_blue_pos = float(light_blue_pos)
        self.yellow_pos = float(yellow_pos)
        super().__init__(vmin, vmax)

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip
        result = np.ma.filled(np.ma.asarray(value), np.nan).astype(float)
        vmin, vmax = self.vmin, self.vmax
        lb, ya = self.light_blue_at, self.yellow_at
        lp, yp = self.light_blue_pos, self.yellow_pos
        den_lb = (lb - vmin) if lb > vmin else 1e-30
        den_mid = (ya - lb) if ya > lb else 1e-30
        den_hi = (vmax - ya) if vmax > ya else 1e-30
        t_lo = (result - vmin) / den_lb * lp
        t_mid = lp + (result - lb) / den_mid * (yp - lp)
        t_hi = yp + (result - ya) / den_hi * (1 - yp)
        t = np.where(result <= lb, t_lo, np.where(result <= ya, t_mid, t_hi))
        if clip:
            t = np.clip(t, 0, 1)
        return np.ma.filled(np.ma.masked_invalid(t), 0)


def plot_matrix_huanghai(
        matrix: np.ndarray,
        figsize: Tuple[int, int] = (12, 10),
        extent: Optional[Tuple[float, float, float, float]] = None,
        show_colorbar: bool = True,
        colorbar_label: str = "重力异常 / (10⁻⁵ m/s²)",
        colorbar_interval: Optional[float] = 5.0,
        vmin_display: Optional[float] = -50.0,
        vmax_display: Optional[float] = 60.0,
        use_figure_cbar: bool = True,
        light_blue_at_value: Optional[float] = 5.0,
        yellow_at_value: Optional[float] = 30.0,
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        figure_title: Optional[str] = None,
        y_origin: str = "upper",
) -> None:
    """
    仅用色块可视化矩阵（无等高线）。use_figure_cbar=True 时按图示色条：-50～60，
    深蓝→蓝→青→绿→黄→橙→红，刻度间隔 5；否则可用 light_blue_at_value / yellow_at_value 分段。
    """
    if matrix.ndim != 2:
        raise ValueError("matrix 必须是二维数组")
    valid = np.isfinite(matrix)
    if not np.any(valid):
        raise ValueError("matrix 中无有效数值")
    if vmin_display is not None and vmax_display is not None:
        vmin, vmax = float(vmin_display), float(vmax_display)
        if vmin >= vmax:
            vmin, vmax = vmin - 1e-9, vmax + 1e-9
    else:
        vmin = np.nanpercentile(matrix, percentile_low)
        vmax = np.nanpercentile(matrix, percentile_high)
        if vmin >= vmax:
            vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)
            if vmin >= vmax:
                vmin, vmax = vmin - 1e-9, vmax + 1e-9
    if use_figure_cbar:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cm = _huanghai_figure_cmap()
        if colorbar_interval is None:
            colorbar_interval = 5.0
    else:
        if light_blue_at_value is not None and yellow_at_value is not None:
            lb = np.clip(float(light_blue_at_value), vmin, vmax)
            ya = np.clip(float(yellow_at_value), lb, vmax)
            if ya <= lb:
                ya = min(vmax, lb + 1e-6)
            norm = _BlueYellowBreakNorm(vmin, vmax, light_blue_at=lb, yellow_at=ya)
        elif yellow_at_value is not None:
            norm = _YellowBreakNorm(vmin, vmax, yellow_at=float(yellow_at_value))
        else:
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cm = _huanghai_diverging_cmap()

    fig, ax = plt.subplots(figsize=figsize)
    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        ex = [lon_min, lon_max, lat_min, lat_max]
        ax.set_xlabel("东经/°")
        ax.set_ylabel("北纬/°")
    else:
        ex = None
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    im = ax.imshow(matrix, cmap=cm, norm=norm, extent=ex, aspect="auto", origin=y_origin)
    if y_origin == "upper":
        ax.invert_yaxis()
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, label=colorbar_label)
        if colorbar_interval is not None:
            from matplotlib.ticker import MultipleLocator
            cbar.ax.yaxis.set_major_locator(MultipleLocator(colorbar_interval))
        if vmin_display is not None and vmax_display is not None:
            cbar.ax.set_ylim(vmin_display, vmax_display)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    if figure_title:
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.text(0.5, 0.02, figure_title, ha="center", fontsize=12)
    else:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def plot_matrix_huanghai_interactive(
        matrix: np.ndarray,
        figsize: Tuple[int, int] = (12, 10),
        extent: Optional[Tuple[float, float, float, float]] = None,
        colorbar_label: str = "重力异常 / (10⁻⁵ m/s²)",
        initial_vmin: Optional[float] = None,
        initial_vmax: Optional[float] = None,
        initial_yellow_at: Optional[float] = None,
        y_origin: str = "upper",
        save_path: Optional[str] = None,
) -> None:
    """
    带滑条的可视化：可手动调节色条范围（vmin、vmax）和黄色起始值（yellow_at），
    实时更新图像；点击「保存」按钮可保存当前视图。
    """
    if matrix.ndim != 2:
        raise ValueError("matrix 必须是二维数组")
    valid = np.isfinite(matrix)
    if not np.any(valid):
        raise ValueError("matrix 中无有效数值")

    data_min, data_max = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    vmin_init = initial_vmin if initial_vmin is not None else float(np.nanpercentile(matrix, 2))
    vmax_init = initial_vmax if initial_vmax is not None else float(np.nanpercentile(matrix, 98))
    if vmin_init >= vmax_init:
        vmin_init, vmax_init = data_min, data_max
    yellow_init = initial_yellow_at
    if yellow_init is None:
        yellow_init = (vmin_init + vmax_init) * 0.5
    yellow_init = np.clip(yellow_init, vmin_init, vmax_init)

    cm = _huanghai_diverging_cmap()
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.32)
    ax = fig.add_axes([0.12, 0.35, 0.65, 0.57])
    if extent is not None:
        ex = list(extent)
        ax.set_xlabel("东经/°")
        ax.set_ylabel("北纬/°")
    else:
        ex = None
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    norm0 = _YellowBreakNorm(vmin_init, vmax_init, yellow_at=yellow_init)
    im = ax.imshow(matrix, cmap=cm, norm=norm0, extent=ex, aspect="auto", origin=y_origin)
    if y_origin == "upper":
        ax.invert_yaxis()
    cbar_ax = fig.add_axes([0.78, 0.35, 0.03, 0.57])
    cbar = fig.colorbar(im, cax=cbar_ax, label=colorbar_label)
    cbar.ax.set_ylim(vmin_init, vmax_init)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # 滑条范围：略宽于数据范围
    pad = max((data_max - data_min) * 0.1, 1e-6)
    sl_min, sl_max = data_min - pad, data_max + pad

    ax_vmin = fig.add_axes([0.2, 0.22, 0.55, 0.02])
    ax_vmax = fig.add_axes([0.2, 0.17, 0.55, 0.02])
    ax_yellow = fig.add_axes([0.2, 0.12, 0.55, 0.02])
    sl_vmin = mpl.widgets.Slider(ax_vmin, "vmin", sl_min, sl_max, valinit=vmin_init)
    sl_vmax = mpl.widgets.Slider(ax_vmax, "vmax", sl_min, sl_max, valinit=vmax_init)
    sl_yellow = mpl.widgets.Slider(ax_yellow, "黄色起始", sl_min, sl_max, valinit=yellow_init)

    def update(_):
        vmin, vmax = sl_vmin.val, sl_vmax.val
        if vmin >= vmax:
            return
        ya = np.clip(sl_yellow.val, vmin, vmax)
        sl_yellow.set_val(ya)
        norm = _YellowBreakNorm(vmin, vmax, yellow_at=ya)
        im.set_norm(norm)
        cbar.ax.set_ylim(vmin, vmax)
        fig.canvas.draw_idle()

    sl_vmin.on_changed(update)
    sl_vmax.on_changed(update)
    sl_yellow.on_changed(update)

    def save_current(event):
        path = save_path
        if not path:
            from datetime import datetime
            path = f"huanghai_interactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"已保存: {path}")

    ax_save = fig.add_axes([0.2, 0.02, 0.2, 0.06])
    btn = mpl.widgets.Button(ax_save, "保存当前视图")
    btn.on_clicked(save_current)
    if save_path:
        fig.text(0.5, 0.01, f"保存路径: {save_path}", ha="center", fontsize=9)

    plt.show()
    plt.close(fig)


def _huanghai_figure_cmap():
    """图中色条：-50～60，深蓝→蓝→青；约 5 为青、约 10 为黄；再黄→橙→红"""
    # 归一化 (v+50)/110：-50→0, -35→0.136, -15→0.318, 0→0.455, 5→0.5, 10→0.545, 25→0.682, 40→0.818, 60→1.0
    return mpl.colors.LinearSegmentedColormap.from_list(
        "huanghai_figure",
        [
            (0.0, "#1a237e"),    # -50 深蓝
            (0.14, "#1565c0"),   # -35 蓝
            (0.32, "#0288d1"),   # -15 蓝青
            (0.455, "#00bcd4"),  # 0 青
            (0.5, "#00acc1"),    # 5 青
            (0.545, "#fdd835"),  # 10 黄
            (0.68, "#ff8f00"),   # 25 橙
            (0.82, "#f4511e"),   # 40 橙红
            (1.0, "#c62828"),    # 60 红
        ],
        N=256,
    )


def _huanghai_diverging_cmap():
    """北黄海风格：深蓝→亮蓝→青→绿→黄→多级橙红；5 为亮蓝，再往下为深蓝"""
    return mpl.colors.LinearSegmentedColormap.from_list(
        "huanghai_diverging",
        [
            "#0d47a1",  # 深蓝（最低值）
            "#42a5f5",  # 亮蓝（约从 5 开始）
            "#1565c0",  # 蓝
            "#00acc1",  # 青
            "#00897b",  # 青绿
            "#43a047",  # 绿（零附近）
            "#7cb342",  # 黄绿
            "#fdd835",  # 黄
            "#ffca28",  # 浅橙黄
            "#ffc107",  # 金黄
            "#ffb300",  # 琥珀
            "#ffa000",  # 橙黄
            "#ff8f00",  # 橙
            "#ff7f00",  # 中橙
            "#ff6f00",  # 深橙
            "#ff5722",  # 橙红
            "#f4511e",  # 深橙红
            "#e64a19",  # 朱红
            "#e53935",  # 红
            "#d84315",  # 深红
            "#c62828",  # 大红
            "#b71c1c",  # 暗红（高值）
        ],
        N=256,
    )


def plot_contour_huanghai(
        matrix: np.ndarray,
        title: str = "data",
        figsize: Tuple[int, int] = (12, 10),
        cmap: Union[str, Any] = "huanghai_diverging",
        levels: Optional[Union[int, List[float]]] = None,
        show_colorbar: bool = False,
        plot_type: str = "filled",
        save_path: Optional[str] = None,
        show_plot: bool = True,
        y_origin: str = "upper",
        aspect: Optional[str] = "equal",
        percentile_low: float = 2.0,
        percentile_high: float = 98.0,
        extent: Optional[Tuple[float, float, float, float]] = None,
        colorbar_label: str = "重力异常 / (10⁻⁵ m/s²)",
        colorbar_interval: Optional[float] = None,
        division_lines: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        annotations: Optional[List[Tuple[float, float, str]]] = None,
        figure_title: Optional[str] = None,
        yellow_at_value: Optional[float] = 15,
        vmin_display: Optional[float] = -20.0,
        vmax_display: Optional[float] = 45.0,
) -> None:
    """
    北黄海及邻区风格重力异常等高线图：发散型配色（蓝→绿→黄→红），可选固定色条范围。
    vmin_display / vmax_display: 色条只标记该范围（如 -20 与 45）；None 则用百分位数。
    yellow_at_value: 从此数值开始显示黄色（以下为蓝→绿，以上为黄→红）；None 则用线性色标。
    extent: (经度最小, 经度最大, 纬度最小, 纬度最大)，给出时坐标轴为东经/北纬。
    division_lines: 分区线列表，每项为 (经度数组, 纬度数组)，在 extent 给定时绘制黑色虚线。
    annotations: 标注列表，每项为 (经度, 纬度, 文字)，在 extent 给定时绘制白底黑字。
    figure_title: 图下方标题，如 "图4-1 北黄海及邻区自由空间重力异常及其分区图"。
    """
    if matrix.ndim != 2:
        raise ValueError("matrix 必须是二维数组")

    valid = np.isfinite(matrix)
    if not np.any(valid):
        raise ValueError("matrix 中无有效数值")
    if vmin_display is not None and vmax_display is not None:
        vmin, vmax = float(vmin_display), float(vmax_display)
        if vmin >= vmax:
            vmin, vmax = vmin - 1e-9, vmax + 1e-9
    else:
        vmin = np.nanpercentile(matrix, percentile_low)
        vmax = np.nanpercentile(matrix, percentile_high)
        if vmin >= vmax:
            vmin = np.nanmin(matrix)
            vmax = np.nanmax(matrix)
            if vmin >= vmax:
                vmin, vmax = vmin - 1e-9, vmax + 1e-9
    if yellow_at_value is not None:
        norm = _YellowBreakNorm(vmin, vmax, yellow_at=float(yellow_at_value))
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if isinstance(cmap, str) and cmap == "huanghai_diverging":
        cm = _huanghai_diverging_cmap()
    elif isinstance(cmap, str) and cmap == "viridis":
        purple_min = (150 / 255.0, 100 / 255.0, 255 / 255.0)
        cm = mpl.colors.LinearSegmentedColormap.from_list(
            "purple_to_red_huanghai",
            [purple_min, "#0033ff", "#00c8ff", "#00ff6a", "#ffe600", "#ff2a00"],
            N=256,
        )
    else:
        try:
            cm = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        except Exception:
            cm = _huanghai_diverging_cmap()

    if levels is None:
        levels_to_use = 20
    else:
        levels_to_use = levels

    ny, nx = matrix.shape
    if extent is not None:
        lon_min, lon_max, lat_min, lat_max = extent
        x = np.linspace(lon_min, lon_max, nx)
        y = np.linspace(lat_max, lat_min, ny)  # 第0行=北
        xlabel, ylabel = "东经/°", "北纬/°"
    else:
        x = np.arange(nx)
        y = np.arange(ny)
        xlabel, ylabel = "X", "Y"
    X, Y = np.meshgrid(x, y)

    if plot_type == '3d':
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, matrix, cmap=cm, norm=norm, linewidth=0, antialiased=True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(colorbar_label)
        if show_colorbar:
            fig.colorbar(surf, shrink=0.5, aspect=5)
        if y_origin == "upper":
            ax.set_ylim(ax.get_ylim()[::-1])
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=figsize)
    if plot_type == 'filled':
        csf = ax.contourf(X, Y, matrix, levels=levels_to_use, cmap=cm, norm=norm)
        cs = ax.contour(X, Y, matrix, levels=csf.levels, colors='k', linewidths=0.8)
        if show_colorbar:
            cbar = fig.colorbar(csf, ax=ax, label=colorbar_label)
            if colorbar_interval is not None:
                from matplotlib.ticker import MultipleLocator
                cbar.ax.yaxis.set_major_locator(MultipleLocator(colorbar_interval))
            if vmin_display is not None and vmax_display is not None:
                cbar.ax.set_ylim(vmin_display, vmax_display)
    elif plot_type == 'contour':
        cs = ax.contour(X, Y, matrix, levels=levels_to_use, cmap=cm, norm=norm, linewidths=0.8)
        if show_colorbar:
            cbar = fig.colorbar(cs, ax=ax, label=colorbar_label)
            if colorbar_interval is not None:
                from matplotlib.ticker import MultipleLocator
                cbar.ax.yaxis.set_major_locator(MultipleLocator(colorbar_interval))
            if vmin_display is not None and vmax_display is not None:
                cbar.ax.set_ylim(vmin_display, vmax_display)
    else:
        raise ValueError(f"不支持的图表类型: {plot_type}")

    if division_lines and extent is not None:
        for lon_arr, lat_arr in division_lines:
            ax.plot(lon_arr, lat_arr, 'k--', linewidth=1.0)
    if annotations and extent is not None:
        for lx, ly, text in annotations:
            ax.annotate(text, (lx, ly), fontsize=9, color='k',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if y_origin == "upper":
        ax.invert_yaxis()
    if aspect:
        ax.set_aspect(aspect, adjustable='box')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    if figure_title:
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.text(0.5, 0.02, figure_title, ha='center', fontsize=12)
    else:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close(fig)


def calculate_tick_interval(data_range):
    """根据数据范围计算合适的刻度间隔"""
    if data_range == 0:
        return 1
    # 确定数据范围的数量级
    magnitude = 10 ** np.floor(np.log10(data_range))
    # 计算初步间隔
    preliminary_interval = data_range / 10
    # 确定合适的间隔（1, 2, 5, 10的倍数）
    intervals = [1, 2, 5, 10]
    interval = magnitude
    for i in intervals:
        if i * magnitude >= preliminary_interval:
            interval = i * magnitude
            break
    return interval


def plot_line_chart(data, title="line chart", x_label="x", y_label="y",
                    line_colors=None, markers=None, line_width=2, marker_size=6,
                    show_grid=True, show_labels=False, fig_size=(12, 6), show=True):
    # 确保data是二维列表
    if not isinstance(data[0], list):
        data = [data]
    # 设置默认颜色和标记
    default_colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    default_markers = ['o', 's', '^', 'D', 'v', '*', 'p', 'h', '8', 'x']
    # 处理颜色参数
    if line_colors is None:
        line_colors = [default_colors[i % len(default_colors)] for i in range(len(data))]
    elif not isinstance(line_colors, list):
        line_colors = [line_colors] * len(data)
    # 处理标记参数
    if markers is None:
        markers = [default_markers[i % len(default_markers)] for i in range(len(data))]
    elif not isinstance(markers, list):
        markers = [markers] * len(data)

    plt.figure(figsize=fig_size)

    # 计算所有数据的最小值和最大值
    all_values = [val for sublist in data for val in sublist]
    if not all_values:
        min_value, max_value = 0, 10
    else:
        min_value = min(all_values)
        max_value = max(all_values)

    # 计算数据范围并确定合适的刻度间隔
    data_range = max_value - min_value
    tick_interval = calculate_tick_interval(data_range)

    # 调整y轴范围，使其是刻度间隔的整数倍
    y_min = tick_interval * np.floor(min_value / tick_interval)
    y_max = tick_interval * np.ceil(max_value / tick_interval) + 0.1*(max_value-min_value)

    # 确保y轴范围至少有两个刻度
    if y_max == y_min:
        y_max += tick_interval

    # 绘制每条线
    for i, series in enumerate(data):
        x = np.arange(1, len(series) + 1)
        plt.plot(x, series, f'{markers[i]}-', color=line_colors[i],
                 linewidth=line_width, markersize=marker_size,
                 label=f'iter {i}' if len(data) > 1 else None)

        if show_labels:
            for j, value in enumerate(series):
                plt.annotate(f'{value}', (x[j], value), textcoords='offset points',
                             xytext=(0, 5), ha='center', fontsize=9)

    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # 设置坐标轴范围
    plt.xlim(0, max(len(s) for s in data) + 1 if data else 2)
    plt.ylim(y_min, y_max)

    # 设置刻度
    plt.xticks(np.arange(0, max(len(s) for s in data) + 2 if data else 3, 2))
    plt.yticks(np.arange(y_min, y_max + tick_interval / 2, tick_interval))

    if show_grid:
        plt.grid(True, linestyle='--', alpha=0.7)

    if len(data) > 1:
        plt.legend()

    plt.tight_layout()
    if show:
        plt.show()
    # plt.savefig("custom_chart.png", dpi=300)  # 保存图表

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