"""
重力异常正演（基于 Harmonica 长方体棱柱）

参考：https://www.fatiando.org/harmonica/latest/user_guide/forward_modelling/prism.html
棱柱定义：(west, east, south, north, bottom, top)，单位：米，笛卡尔坐标。
球体：均匀球在外部等价于质心处点质量，用 point_gravity 计算。
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
    outline_polygons=None,
    spheres=None,
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
    outline_polygons : list, optional
        用于在图上叠加地质体轮廓，每项为 (N,2) 的数组或 (x_list, y_list)，表示闭合多边形顶点（米）。
    spheres : list, optional
        球体列表，与棱柱场叠加。每项为 (easting, northing, depth, radius, density)：
        easting/northing 球心水平坐标 (m)，depth 埋深 (m，正=参考面以下)，radius 半径 (m)，density (kg/m³)。

    返回
    -----
    coordinates : tuple
        (easting, northing, upward) 观测点坐标。
    result : ndarray
        正演得到的重力场（与 field 对应），与网格同形状。
    """
    # 统一为多棱柱格式（允许无棱柱、仅球体）
    if not prism:
        prisms = []
        densities = []
    elif isinstance(prism[0], (int, float)):
        prisms = [list(prism)]
        densities = [float(density)] if np.isscalar(density) else list(density)
    else:
        prisms = [list(p) for p in prism]
        densities = list(density) if not np.isscalar(density) else [float(density)] * len(prisms)

    # 确定观测区域
    if region is None:
        if prisms:
            all_w = min(p[0] for p in prisms)
            all_e = max(p[1] for p in prisms)
            all_s = min(p[2] for p in prisms)
            all_n = max(p[3] for p in prisms)
            pad_x = (all_e - all_w) * 0.2
            pad_y = (all_n - all_s) * 0.2
            region = (all_w - pad_x, all_e + pad_x, all_s - pad_y, all_n + pad_y)
        elif spheres:
            eastings = [s[0] for s in spheres]
            northings = [s[1] for s in spheres]
            max_r = max(s[3] for s in spheres)
            pad = max(2 * max_r, 1000)
            region = (min(eastings) - pad, max(eastings) + pad, min(northings) - pad, max(northings) + pad)
        else:
            raise ValueError("请提供 prism 或 spheres，或显式指定 region")

    # 观测点网格（verde 与 harmonica 常用方式）
    coordinates = vd.grid_coordinates(region, shape=shape, extra_coords=height)

    # 正演：棱柱
    if prisms:
        result = hm.prism_gravity(coordinates, prisms, densities, field=field)
    else:
        result = np.zeros(coordinates[0].shape)

    # 正演：球体（等价为质心处点质量，质量 = (4/3)*π*R³*ρ）
    if spheres:
        eastings = [s[0] for s in spheres]
        northings = [s[1] for s in spheres]
        upward = [-float(s[2]) for s in spheres]  # depth 为正表示向下，质心 upward 为负
        points = (eastings, northings, upward)
        masses = [(4.0 / 3.0) * np.pi * (float(s[3]) ** 3) * float(s[4]) for s in spheres]
        result = result + hm.point_gravity(
            coordinates, points, masses, field=field, coordinate_system="cartesian"
        )

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
        outline_polygons=outline_polygons,
    )

    # 若有棱柱，额外保存地下异常 3D 可视化图（xy 范围与 region 一致）
    if prisms:
        base, ext = os.path.splitext(os.path.basename(image_path))
        image_3d_path = base + "_3d" + ext
        _plot_prisms_3d(prisms, image_3d_path, output_dir=out, region=region)
        # 同时保存原始长方体分布图（俯视）
        image_layout_path = base + "_layout" + ext
        _plot_prisms_layout(prisms, image_layout_path, output_dir=out, region=region)

    return coordinates, result


def _plot_and_save(coordinates, data, field, image_path, add_contours=False, cmap=None, outline_polygons=None):
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
    # 叠加地质体轮廓（闭合多边形）
    if outline_polygons:
        for poly in outline_polygons:
            if hasattr(poly, "__len__") and len(poly) == 2 and hasattr(poly[0], "__len__"):
                xp, yp = np.asarray(poly[0]), np.asarray(poly[1])
            else:
                arr = np.asarray(poly)
                xp, yp = arr[:, 0], arr[:, 1]
            if xp[0] != xp[-1] or yp[0] != yp[-1]:
                xp = np.append(xp, xp[0])
                yp = np.append(yp, yp[0])
            ax.plot(xp, yp, "k-", linewidth=1.2, alpha=0.9)
        # 指北箭头（Y 轴为北）
        ax.annotate("", xy=(0.92, 0.96), xycoords="axes fraction", xytext=(0.92, 0.88),
                    arrowprops=dict(arrowstyle="->", color="k", lw=1.5))
        ax.annotate("N", xy=(0.92, 0.98), xycoords="axes fraction", fontsize=12, fontweight="bold", ha="center")
    ax.set_aspect("equal")
    # 坐标轴用普通数字显示，不用科学计数法
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    cbar = plt.colorbar(im, ax=ax, label=unit)
    cbar.ax.ticklabel_format(style="plain", useOffset=False)
    plt.tight_layout()
    plt.savefig(image_path, dpi=150)
    plt.close()
    print(f"正演图像已保存: {image_path}")


def _prism_to_3d_faces(prism):
    """
    将单个棱柱 (west, east, south, north, bottom, top) 转为 3D 六个面的顶点列表。
    深度轴：深度 (m) = -upward，即正值向下，与地质习惯一致。
    """
    w, e, s, n, bottom, top = prism
    z_top = -float(top)    # 顶面深度 (m)，较小
    z_bottom = -float(bottom)  # 底面深度 (m)，较大
    # 六个面：底面、顶面、前(y=s)、后(y=n)、左(x=w)、右(x=e)
    faces = [
        [(w, s, z_bottom), (e, s, z_bottom), (e, n, z_bottom), (w, n, z_bottom)],  # 底面
        [(w, s, z_top), (e, s, z_top), (e, n, z_top), (w, n, z_top)],              # 顶面
        [(w, s, z_bottom), (e, s, z_bottom), (e, s, z_top), (w, s, z_top)],        # 前 y=s
        [(w, n, z_bottom), (e, n, z_bottom), (e, n, z_top), (w, n, z_top)],        # 后 y=n
        [(w, s, z_bottom), (w, n, z_bottom), (w, n, z_top), (w, s, z_top)],        # 左 x=w
        [(e, s, z_bottom), (e, n, z_bottom), (e, n, z_top), (e, s, z_top)],        # 右 x=e
    ]
    return faces


def _plot_prisms_3d(prisms, image_path, colors=None, alpha=0.4, output_dir=None, grid_size=200, region=None):
    """
    将地下异常棱柱绘制为 3D 半透明长方体并保存。
    坐标轴：Easting (m)、Northing (m)、Depth (m, 向下为正)。
    grid_size：坐标轴网格间距 (m)，三轴统一，默认 200。
    region：tuple (west, east, south, north)，与正演一致；若提供则 xy 轴范围使用 region，否则由棱柱范围外扩。
    """
    if not prisms:
        return
    out = output_dir if output_dir is not None else OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    save_path = os.path.join(out, os.path.basename(image_path))

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(prisms))]
    elif len(colors) < len(prisms):
        colors = list(colors) + [colors[-1]] * (len(prisms) - len(colors))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, prism in enumerate(prisms):
        faces = _prism_to_3d_faces(prism)
        poly = Poly3DCollection(
            faces,
            facecolor=colors[i],
            edgecolor=colors[i],
            linewidths=0.8,
            alpha=alpha,
        )
        ax.add_collection3d(poly)

    # 设置坐标范围：xy 与 region 对应，z 由棱柱深度范围决定，并对齐到 grid_size
    g = grid_size
    # 这里的 z 轴使用“深度”(m)，向下为正：top/bottom 通常为负值（upward），取相反数得到深度
    min_depth = min(-float(p[5]) for p in prisms)      # 顶面最浅深度
    max_depth = max(-float(p[4]) for p in prisms)      # 底面最深深度
    if region is not None:
        west, east, south, north = region
        x_min = np.floor(west / g) * g
        x_max = np.ceil(east / g) * g
        y_min = np.floor(south / g) * g
        y_max = np.ceil(north / g) * g
    else:
        all_w = min(p[0] for p in prisms)
        all_e = max(p[1] for p in prisms)
        all_s = min(p[2] for p in prisms)
        all_n = max(p[3] for p in prisms)
        pad_x = max((all_e - all_w) * 0.1, g)
        pad_y = max((all_n - all_s) * 0.1, g)
        x_min = np.floor((all_w - pad_x) / g) * g
        x_max = np.ceil((all_e + pad_x) / g) * g
        y_min = np.floor((all_s - pad_y) / g) * g
        y_max = np.ceil((all_n + pad_y) / g) * g
    pad_z = max((max_depth - min_depth) * 0.1, g)
    # 深度轴从 0 或更浅处开始更直观（避免出现负深度刻度）
    z_min = max(0.0, np.floor((min_depth - pad_z) / g) * g)
    z_max = np.ceil((max_depth + pad_z) / g) * g
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.invert_zaxis()  # 深度向下为正：视觉上“向下更深”

    # 根据各轴范围自动分配刻度间距，避免过密或过疏
    def _auto_tick_step(span, target_ticks=7):
        span = float(max(span, 1.0))
        raw = span / max(target_ticks, 2)
        mag = 10 ** np.floor(np.log10(raw))
        norm = raw / mag
        if norm <= 1:
            base = 1
        elif norm <= 2:
            base = 2
        elif norm <= 2.5:
            base = 2.5
        elif norm <= 5:
            base = 5
        else:
            base = 10
        return base * mag

    step_x = _auto_tick_step(x_max - x_min, target_ticks=7)
    step_y = _auto_tick_step(y_max - y_min, target_ticks=7)
    step_z = _auto_tick_step(z_max - z_min, target_ticks=6)
    ax.xaxis.set_major_locator(MultipleLocator(step_x))
    ax.yaxis.set_major_locator(MultipleLocator(step_y))
    ax.zaxis.set_major_locator(MultipleLocator(step_z))

    # 等比例显示：各轴单位长度一致，网格在视觉上大小统一
    range_x = x_max - x_min
    range_y = y_max - y_min
    range_z = z_max - z_min
    ax.set_box_aspect((range_x, range_y, range_z))

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_zlabel("Depth (m)")
    ax.view_init(elev=20, azim=-55)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"地下异常 3D 图已保存: {save_path}")


def _plot_prisms_layout(prisms, image_path, output_dir=None, region=None):
    """
    绘制原始长方体分布俯视图（XY 平面）并保存。
    每个棱柱按其水平投影矩形显示，便于核对模型几何分布。
    """
    if not prisms:
        return
    out = output_dir if output_dir is not None else OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    save_path = os.path.join(out, os.path.basename(image_path))

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("tab10")

    for i, p in enumerate(prisms):
        w, e, s, n = p[0], p[1], p[2], p[3]
        color = cmap(i % 10)
        xs = [w, e, e, w, w]
        ys = [s, s, n, n, s]
        ax.fill(xs, ys, facecolor=color, edgecolor=color, alpha=0.25)
        ax.plot(xs, ys, color=color, linewidth=1.2)
        # ax.text((w + e) * 0.5, (s + n) * 0.5, f"{i+1}", color=color, fontsize=9, ha="center", va="center")

    if region is not None:
        west, east, south, north = region
    else:
        west = min(p[0] for p in prisms)
        east = max(p[1] for p in prisms)
        south = min(p[2] for p in prisms)
        north = max(p[3] for p in prisms)
        pad_x = (east - west) * 0.1 if east > west else 100.0
        pad_y = (north - south) * 0.1 if north > south else 100.0
        west, east = west - pad_x, east + pad_x
        south, north = south - pad_y, north + pad_y

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"原始长方体分布图已保存: {save_path}")


def run_four_bodies_example():
    """
    仿照参考图：四个地质体 B1、B2、B3、B4 的重力异常正演。
    B1 右上角方形；B2 中下梯形；B3 中左梯形；B4 北西-南东向条带状（岩墙/脉体）。
    坐标系：X、Y 约 0–9000 m，棱柱用矩形近似多边形，图上叠加地质体轮廓。
    """
    # B1：右上角方形 (west, east, south, north, bottom, top)，单位 m
    b1 = [7200, 8200, 7200, 8200, -2000, -500]
    # B2：中下梯形，用 3 个棱柱近似
    b2a = [2500, 3500, 1000, 3800, -2000, -500]
    b2b = [3500, 5800, 1000, 3800, -2000, -500]
    b2c = [5800, 6000, 1000, 3800, -2000, -500]
    # B3：中左梯形，用 2 个棱柱近似
    b3a = [1500, 2750, 5500, 7500, -2000, -500]
    b3b = [2750, 4000, 5500, 7500, -2000, -500]
    # B4：北西-南东向条带（弯曲），用 6 个棱柱沿走向近似
    b4_boxes = [
        [500, 2500, 7000, 9000, -2000, -500],
        [2500, 4500, 5500, 7500, -2000, -500],
        [4500, 6500, 4000, 6000, -2000, -500],
        [6500, 8500, 3000, 5000, -2000, -500],
        [8500, 9000, 2500, 3500, -2000, -500],
        [5500, 7500, 3500, 5500, -2000, -500],  # 衔接弯曲中段
    ]

    prisms = [b1, b2a, b2b, b2c, b3a, b3b] + b4_boxes
    densities = [3000] * len(prisms)  # kg/m³，可改为分体赋不同密度

    # 地质体轮廓多边形（用于在图上勾边），顶点顺序闭合
    outline_b1 = np.array([[7200, 7200], [8200, 7200], [8200, 8200], [7200, 8200]])
    outline_b2 = np.array([[2500, 1000], [5800, 1000], [6000, 3800], [3500, 3800]])
    outline_b3 = np.array([[1500, 5500], [3500, 5500], [4000, 7500], [1800, 7500]])
    outline_b4 = np.array([
        [500, 9000], [4000, 7000], [5500, 5000], [8500, 3500], [9000, 2500],
        [6000, 4000], [4500, 6000], [1000, 8000],
    ])
    outline_polygons = [outline_b1, outline_b2, outline_b3, outline_b4]

    region = (0, 9000, 0, 9000)
    coordinates, g_z = prism_gravity_forward(
        prism=prisms,
        density=densities,
        region=region,
        shape=(91, 91),
        height=0.0,
        field="g_z",
        image_path="gravity_four_bodies.png",
        npy_path="gravity_four_bodies.npy",
        add_contours=True,
        cmap="viridis",
        output_dir=OUTPUT_DIR,
        outline_polygons=outline_polygons,
    )
    return coordinates, g_z


def run_sphere_example():
    """
    球体重力正演示例：可仅用球体，或棱柱+球体叠加。
    球体参数 (easting, northing, depth, radius, density)，depth 为埋深 (m)。
    """
    # 球体：(东, 北, 埋深, 半径, 密度)，单位 m、kg/m³
    spheres = [
        (4500, 5000, 600, 350, 3200),   # 中部偏左高密度球
        (7000, 3000, 800, 250, 3100),   # 东南侧球
    ]

    coordinates, g_z = prism_gravity_forward(
        prism=[],
        density=[],
        spheres=spheres,
        region=(0, 9000, 0, 9000),
        shape=(91, 91),
        height=0.0,
        field="g_z",
        image_path="gravity_sphere.png",
        npy_path="gravity_sphere.npy",
        add_contours=True,
        cmap="viridis",
        output_dir=OUTPUT_DIR,
    )
    return coordinates, g_z



def run_cuboid_example():
    """
    单一 长方体 正演
    """
    # 棱柱 (西, 东, 南, 北, 底, 顶)，单位 m
    prisms = [
        [1500, 8500 , 4500, 5500, -2500 , 2500],
    ]
    # 密度 kg/m³：基底 / 矿化体×2 / 岩墙 / 东南异常 / 凹陷（相对低）
    densities = [3300]

    region = (0, 10000, 0, 10000)
    coordinates, g_z = prism_gravity_forward(
        prism=prisms,
        density=densities,
        region=region,
        shape=(100, 100),
        height=20.0,
        field="g_z",
        image_path="gravity_forward_cuboid.png",
        npy_path="gravity_forward_cuboid.npy",
        add_contours=True,
        cmap="RdYlBu_r",  # 红-黄-蓝，便于区分正负异常
        output_dir=OUTPUT_DIR,
    )
    return coordinates, g_z



def run_two_cuboid_example():
    """
    两个 长方体 排列 正演
    """
    # 棱柱 (西, 东, 南, 北, 底, 顶)，单位 m
    prisms = [
        [2000, 4000 , 2000, 8000, -2500 , -500],
        [6000, 8000 , 2000, 8000, -2500 , -500]
    ]

    densities = [3300, 3300 ]

    region = (0, 10000, 0, 10000)
    coordinates, g_z = prism_gravity_forward(
        prism=prisms,
        density=densities,
        region=region,
        shape=(100, 100),
        height=20.0,
        field="g_z",
        image_path="gravity_forward_two_cuboid.png",
        npy_path="gravity_forward_two_cuboid.npy",
        add_contours=True,
        cmap="RdYlBu_r",  # 红-黄-蓝，便于区分正负异常
        output_dir=OUTPUT_DIR,
    )
    return coordinates, g_z




def run_complex_example():
    """
    较复杂的重力异常正演示例：多个不同深度、尺度与密度的棱柱，
    模拟基底隆起、局部高密度体、线性构造等叠加效应。
    区域为 (0, 10000, 0, 10000)，单位 m。
    """
    # 棱柱 (西, 东, 南, 北, 底, 顶)，单位 m，全部落在 [0,10000]×[0,10000]
    prisms = [

        [0, 3000, 9000, 9500, -2500, -500],
        [2500, 3000, 6500, 9500, -2500, -500],
        [3000, 5500, 6500, 7000, -2500, -500],
        [5000, 5500, 4000, 6500, -2500, -500],
        [5500, 8000, 4000, 4500, -2500, -500],
        [7500, 8000, 1500, 4000, -2500, -500],
        [8000, 10000, 1500, 2000, -2500, -500],

        [1000, 3000, 1000, 2500, -2500, -500],
        [2000, 3000, 2500, 3500, -2500, -500],

        [7000, 8000, 7500, 9000, -2500, -500],

    ]

    densities = [3200, 3600, 3600, 3400, 3800, 3800,3800,    2800,2800,3400]

    region = (0, 10000, 0, 10000)
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




def run_line_example():
    """
    较复杂的重力异常正演示例：多个不同深度、尺度与密度的棱柱，
    模拟基底隆起、局部高密度体、线性构造等叠加效应。
    区域为 (0, 10000, 0, 10000)，单位 m。
    """
    # 棱柱 (西, 东, 南, 北, 底, 顶)，单位 m，全部落在 [0,10000]×[0,10000]
    prisms = [

        [300, 700, 0, 1000, -400, -200],
        # [0, 600, 100, 500, -400, -200],
        
    ]

    densities = [1000]

    region = (-200, 1200, -200, 1200)
    coordinates, g_z = prism_gravity_forward(
        prism=prisms,
        density=densities,
        region=region,
        shape=(120, 120),
        height=20.0,
        field="g_z",
        image_path="gravity_forward_line.png",
        npy_path="gravity_forward_line.npy",
        add_contours=True,
        cmap="RdYlBu_r",  # 红-黄-蓝，便于区分正负异常
        output_dir=OUTPUT_DIR,
    )
    return coordinates, g_z




if __name__ == "__main__":
    # 运行四地质体（B1–B4）正演示例，与参考图类似
    # coordinates, g_z = run_four_bodies_example()
    # # 运行球体正演示例（棱柱+球体可同时用 spheres 参数）
    coordinates, g_z = run_complex_example()
    coordinates, g_z = run_line_example()
    coordinates, g_z = run_two_cuboid_example()


    coordinates, g_z = run_cuboid_example()
