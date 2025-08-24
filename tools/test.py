from scipy.ndimage import map_coordinates
from utils.utils import *
import os

def calculate_tangent_direction(matrix):
    """计算矩阵每个点的切线方向（最大斜率方向的切线）和斜率大小"""
    # 计算x和y方向的一阶偏导数
    dx, dy = np.gradient(matrix)

    # 梯度大小（斜率强度）
    grad_magnitude = np.sqrt(dx**2 + dy**2)

    # 梯度方向（法向量方向）
    grad_direction = np.arctan2(dy, dx)

    # 切线方向 = 梯度方向 + 90度
    tangent_direction = grad_direction + np.pi / 2

    # 标准化角度到[-π, π]范围
    tangent_direction = (tangent_direction + np.pi) % (2 * np.pi) - np.pi

    return tangent_direction, grad_magnitude

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

    # 坐标网格 - 注意：先y后x，因为矩阵是[行,列]
    y_coords, x_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")

    for k in range(-half, half+1):
        # 计算每个采样点相对中心点的偏移
        dx = k * step * np.cos(tangent_direction)  # x方向偏移
        dy = k * step * np.sin(tangent_direction)  # y方向偏移

        # 新坐标 - 注意：矩阵坐标是[y, x]
        sample_x = x_coords + dx
        sample_y = y_coords + dy

        # 插值获取采样值（使用线性插值）
        values = map_coordinates(matrix, [sample_y.ravel(), sample_x.ravel()],
                                 order=1, mode='nearest').reshape(rows, cols)

        samples[:, :, k+half] = values

    return samples


def plot_tangent(matrix, directions, magnitudes, save_path, show_plot=False):
    """可视化切线方向和斜率大小"""
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
    norm = plt.cm.colors.Normalize(vmin=np.min(grad_mag), vmax=np.max(grad_mag))

    plt.quiver(X, Y, U, V, grad_mag, cmap='jet', norm=norm,
               scale=50, width=0.002, headwidth=3)

    plt.colorbar(label='Slope Magnitude')
    plt.title('Tangent Directions (Max Slope)')
    plt.axis('off')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show_plot:
        plt.show()
    plt.close()

def visualize_sampling(matrix, tangent_direction,slope_magnitude, samples, step=1.0, skip=5):
    """
    可视化采样点直线
    :param matrix: 原始矩阵
    :param tangent_direction: 切线方向矩阵
    :param samples: 采样结果
    :param step: 采样间隔
    :param skip: 每隔多少点绘制一条线（避免过于密集）
    """
    rows, cols = matrix.shape
    num_points = samples.shape[2]
    half = num_points // 2

    # 创建画布
    plt.figure(figsize=(12, 10))

    # 绘制原始矩阵作为背景
    plt.imshow(matrix, cmap='gray', alpha=0.7, origin='upper')  # 明确设置原点位置

    # 只绘制部分点的采样线（避免过于密集）
    for i in range(0, rows, skip):
        for j in range(0, cols, skip):
            if slope_magnitude[i][j] == 0:
                continue
            # 计算该点的所有采样点坐标
            x_coords = []
            y_coords = []
            # y_coords1, x_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
            for k in range(-half, half+1):
                angle = tangent_direction[i, j]
                if angle > 1.6:
                    print("----")
                dx = k * step * np.cos(angle)  # x方向偏移
                dy = - k * step * np.sin(angle)  # y方向偏移

                # 注意：j是x坐标，i是y坐标
                x_coords.append(j + dx)
                y_coords.append(i + dy)

            # 绘制采样点连成的直线
            plt.plot(x_coords, y_coords, 'r-', linewidth=1, alpha=0.6)
            # 标记中心点
            plt.scatter(j, i, c='blue', s=10, alpha=0.8)

    plt.title('采样点直线可视化（红色为采样线，蓝色为中心点）')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# 生成示例数据
if __name__ == "__main__":
    # 创建一个测试矩阵（含半圆形区域）
    size = 20  # 增大尺寸以便更好地观察
    matrix = np.zeros((size, size))
    x = np.arange(size)
    y = np.arange(size)
    xx, yy = np.meshgrid(x, y)

    # 半圆参数
    center_x = size // 2
    center_y = size // 2
    radius = size // 3

    # 创建上半圆 (y <= center_y)
    # 条件1: 在圆内
    in_circle = (xx - center_x)**2 + (yy - center_y)** 2 < radius**2
    # 条件2: 在上半圆 (y坐标小于等于圆心y坐标)
    upper_half = yy <= center_y
    # 合并条件得到半圆
    semi_circle = in_circle & upper_half

    matrix[semi_circle] = 1.0

    # 创建输出目录
    output_dir = os.path.join("output")
    os.makedirs(output_dir, exist_ok=True)


    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Value")
    plt.title("Heatmap with matplotlib")
    plt.show()


    # 计算切线方向
    tangent_direction, slope_magnitude = calculate_tangent_direction(matrix)

    plot_tangent(matrix, tangent_direction, slope_magnitude,save_path=None,show_plot=True)

    # 沿切线方向采样
    num_points = 5  # 5个采样点（中心±2）
    samples = sample_along_tangent(matrix, tangent_direction, num_points=num_points, step=1)

    # 可视化结果，设置skip=3以便更清晰地展示
    visualize_sampling(matrix, tangent_direction, slope_magnitude, samples, step=1, skip=1)
