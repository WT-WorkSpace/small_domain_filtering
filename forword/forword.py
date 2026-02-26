"""
重力异常正演（基于 Harmonica 长方体棱柱）

参考：https://www.fatiando.org/harmonica/latest/user_guide/forward_modelling/prism.html
棱柱定义：(west, east, south, north, bottom, top)，单位：米，笛卡尔坐标。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import harmonica as hm
import verde as vd

# 默认输出目录：所有正演图像与 npy 数据保存于此
OUTPUT_DIR = "forward_output"


def prism_gravity_forward(
    prism,
    density,
    region=None,
    shape=(51, 51),
    height=0.0,
    field="g_z",
    image_path="gravity_forward.png",
    npy_path="gravity_forward.npy",
    add_contours=False,
    cmap=None,
    output_dir=None,
):
    """
    单一体或多数长方体的重力异常正演。

    参数
    -----
    prism : tuple/list 或 list of list
        单个棱柱：(west, east, south, north, bottom, top)，单位 m；
        或多个棱柱：列表，每个元素为 (w, e, s, n, bottom, top)。
    density : float 或 array-like
        密度 (kg/m³)。单一体为标量；多体为与棱柱数量一致的列表/数组。
    region : tuple, optional
        观测网格范围 (west, east, south, north)，单位 m。若为 None，从棱柱水平范围外扩 20%。
    shape : tuple
        观测点网格形状 (nx, ny)，默认 (51, 51)。
    height : float
        观测面高度 (m)，默认 0。
    field : str
        重力场分量，如 "g_z"(垂向)、"potential"、"g_e"、"g_n" 等，默认 "g_z"。
    image_path : str
        正演结果图像保存路径，默认 "gravity_forward.png"。
    npy_path : str
        正演结果数组保存路径（numpy .npy），默认 "gravity_forward.npy"。
    add_contours : bool
        是否叠加等值线，默认 False。
    cmap : str, optional
        色标名称，默认 "viridis"；重力异常可用 "RdYlBu_r" 等。
    output_dir : str, optional
        输出目录，若给定则在该目录下创建并保存 image_path 与 npy_path；默认使用 OUTPUT_DIR。

    返回
    -----
    coordinates : tuple
        (easting, northing, upward) 观测点坐标。
    result : ndarray
        正演得到的重力场（与 field 对应），与网格同形状。
    """
    # 统一为多棱柱格式
    if isinstance(prism[0], (int, float)):
        prisms = [list(prism)]
        densities = [float(density)] if np.isscalar(density) else list(density)
    else:
        prisms = [list(p) for p in prism]
        densities = list(density) if not np.isscalar(density) else [float(density)] * len(prisms)

    # 确定观测区域
    if region is None:
        all_w = min(p[0] for p in prisms)
        all_e = max(p[1] for p in prisms)
        all_s = min(p[2] for p in prisms)
        all_n = max(p[3] for p in prisms)
        pad_x = (all_e - all_w) * 0.2
        pad_y = (all_n - all_s) * 0.2
        region = (all_w - pad_x, all_e + pad_x, all_s - pad_y, all_n + pad_y)

    # 观测点网格（verde 与 harmonica 常用方式）
    coordinates = vd.grid_coordinates(region, shape=shape, extra_coords=height)

    # 正演
    result = hm.prism_gravity(coordinates, prisms, densities, field=field)

    # 统一保存到输出目录
    out = output_dir if output_dir is not None else OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    save_image = os.path.join(out, os.path.basename(image_path))
    save_npy = os.path.join(out, os.path.basename(npy_path))

    # 保存为 numpy
    np.save(save_npy, result)
    print(f"正演结果已保存: {save_npy}")

    # 绘图并保存
    _plot_and_save(
        coordinates, result, field, save_image,
        add_contours=add_contours, cmap=cmap,
    )

    return coordinates, result


def _plot_and_save(coordinates, data, field, image_path, add_contours=False, cmap=None):
    """绘制正演场并保存图像。"""
    easting, northing = coordinates[0], coordinates[1]
    if field == "potential":
        unit = "J/kg"
    elif field in ("g_ee", "g_nn", "g_zz", "g_en", "g_ez", "g_nz"):
        unit = "Eotvos"
    else:
        unit = "mGal"
    if cmap is None:
        cmap = "viridis"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(easting, northing, data, shading="auto", cmap=cmap)
    if add_contours:
        levels = np.linspace(data.min(), data.max(), 15)
        ax.contour(easting, northing, data, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
    ax.set_aspect("equal")
    # 坐标轴用普通数字显示，不用科学计数法
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    cbar = plt.colorbar(im, ax=ax, label=unit)
    cbar.ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_title(f"重力正演 - {field}")
    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    plt.close()
    print(f"正演图像已保存: {image_path}")


def run_complex_example():
    """
    较复杂的重力异常正演示例：多个不同深度、尺度与密度的棱柱，
    模拟基底隆起、局部高密度体、线性构造等叠加效应。
    """
    # 棱柱 (西, 东, 南, 北, 底, 顶)，单位 m
    prisms = [
        # 1. 深部大范围基底隆起（中等密度）
        [-8000, 8000, -6000, 6000, -5000, -2500],
        # 2. 浅部高密度矿化体（偏北东）
        [1000, 4000, 2000, 5000, -1200, -400],
        # 3. 另一浅部高密度体（西南）
        [-5000, -2000, -4000, -1000, -800, -200],
        # 4. 北西向条带状异常（似断裂/岩墙）
        [-3000, -1500, 3000, 5000, -2000, -600],
        # 5. 小尺度局部异常（东南角）
        [5000, 7000, -5000, -3000, -1500, -700],
        # 6. 中部低密度凹陷（密度低于围岩，用较低密度表示相对亏损）
        [-2000, 0, -1000, 1000, -2500, -1200],
    ]
    # 密度 kg/m³：基底 / 矿化体×2 / 岩墙 / 东南异常 / 凹陷（相对低）
    densities = [2850, 3400, 3200, 3100, 3050, 2550]

    region = (-12e3, 12e3, -10e3, 10e3)
    coordinates, g_z = prism_gravity_forward(
        prism=prisms,
        density=densities,
        region=region,
        shape=(120, 120),
        height=20.0,
        field="g_z",
        image_path="gravity_forward_complex.png",
        npy_path="gravity_forward_complex.npy",
        add_contours=True,
        cmap="RdYlBu_r",  # 红-黄-蓝，便于区分正负异常
        output_dir=OUTPUT_DIR,
    )
    return coordinates, g_z


if __name__ == "__main__":
    # 运行较复杂的多棱柱正演示例
    coordinates, g_z = run_complex_example()
