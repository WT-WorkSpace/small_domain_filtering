# 重力异常正演

基于 [Harmonica](https://www.fatiando.org/harmonica/latest/user_guide/forward_modelling/prism.html) 的长方体棱柱重力正演：给定棱柱坐标与密度，计算观测面上的重力场并出图、保存为 NumPy 数组。

## 棱柱定义

- 每个长方体由 **6 个边界** 表示：`(west, east, south, north, bottom, top)`。
- 单位：**米**；坐标系：**笛卡尔**（适用于小范围区域，忽略地球曲率）。
- 可同时传入多个棱柱及各自密度，正演结果为叠加场。

## 依赖与安装

```bash
cd forword
pip install -r requirements.txt
```

依赖：`numpy`、`matplotlib`、`harmonica`、`verde`。

## 输出目录

- 所有正演得到的 **PNG 图像** 和 **.npy 数据** 默认保存到 **`forward_output/`** 目录（不存在时自动创建）。
- 可通过参数 `output_dir` 指定其他目录。

## 快速运行

直接运行脚本会执行内置的“复杂示例”（多棱柱、等值线、红-黄-蓝色标）：

```bash
python forword.py
```

结果将出现在 `forward_output/` 下，例如：
- `forward_output/gravity_forward_complex.png`
- `forward_output/gravity_forward_complex.npy`

## 用法说明

### 主要函数

- **`prism_gravity_forward(prism, density, ...)`**：单一体或多数长方体的重力正演，返回观测坐标与场值数组，并将图像与数组写入 `output_dir`（默认 `forward_output/`）。

### 单棱柱示例

```python
from forword import prism_gravity_forward, OUTPUT_DIR
import numpy as np

# 棱柱 (西, 东, 南, 北, 底, 顶)，单位 m
prism = (-2000, 2000, -2000, 2000, -1600, -900)
density = 3300  # kg/m³

coordinates, g_z = prism_gravity_forward(
    prism=prism,
    density=density,
    region=(-10000, 10000, -10000, 10000),
    shape=(51, 51),
    height=10.0,
    field="g_z",
    image_path="gravity_forward.png",
    npy_path="gravity_forward.npy",
    output_dir=OUTPUT_DIR,  # 可选，不传则用默认 forward_output
)

# 读回保存的数组（路径在输出目录下）
data = np.load("forward_output/gravity_forward.npy")
```

### 多棱柱示例

```python
prisms = [
    [-2000, 2000, -2000, 2000, -3000, -1000],
    [3000, 5000, 2000, 4000, -2500, -500],
]
densities = [2670, 3300]

coordinates, g_z = prism_gravity_forward(
    prism=prisms,
    density=densities,
    region=(0, 10000, 0, 10000),
    shape=(80, 80),
    image_path="gravity_multi.png",
    npy_path="gravity_multi.npy",
)
# 文件保存在 forward_output/ 下
```

### 添加球体

均匀球体在观测点外部产生的重力场等价于**质心处同质量点质量**，通过参数 `spheres` 传入，可与棱柱同时使用。

每个球体为元组 **`(easting, northing, depth, radius, density)`**：

- `easting`, `northing`：球心水平坐标 (m)
- `depth`：埋深 (m)，**正数表示参考面以下**
- `radius`：半径 (m)
- `density`：密度 (kg/m³)

示例：一个球心在 (5000, 5000)、埋深 800 m、半径 400 m、密度 3200 kg/m³ 的球体，与棱柱叠加正演：

```python
from forword import prism_gravity_forward, OUTPUT_DIR

prism = [0, 10000, 0, 10000, -2000, -500]   # 一层板
density = 2700
# 球体：(东, 北, 埋深, 半径, 密度)
spheres = [(5000, 5000, 800, 400, 3200)]

coordinates, g_z = prism_gravity_forward(
    prism=prism,
    density=density,
    spheres=spheres,
    region=(0, 10000, 0, 10000),
    shape=(80, 80),
    image_path="gravity_with_sphere.png",
    npy_path="gravity_with_sphere.npy",
    output_dir=OUTPUT_DIR,
)
```

仅球体、无棱柱时，传 `prism=[]` 并指定 `region` 即可。

### 常用参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `prism` | 单棱柱 tuple 或多棱柱 list | — |
| `density` | 密度 (kg/m³)，标量或与棱柱同长的 list | — |
| `region` | 观测范围 (west, east, south, north)，单位 m | 由棱柱外扩 20% |
| `shape` | 网格点数 (nx, ny) | (51, 51) |
| `height` | 观测面高度 (m) | 0 |
| `field` | 场分量，见下表 | "g_z" |
| `image_path` | 图像文件名（存于 output_dir） | "gravity_forward.png" |
| `npy_path` | .npy 文件名（存于 output_dir） | "gravity_forward.npy" |
| `add_contours` | 是否叠加等值线 | False |
| `cmap` | 色标名称（如 "viridis", "RdYlBu_r"） | "viridis" |
| `output_dir` | 输出目录 | `OUTPUT_DIR`（即 "forward_output"） |
| `outline_polygons` | 图上叠加轮廓多边形列表 | None |
| `spheres` | 球体列表，每项 (easting, northing, depth, radius, density) | None |

### 可选场分量 `field`

- `"g_z"`：垂向重力加速度（常用），单位 mGal。
- `"potential"`：重力位，单位 J/kg。
- `"g_e"`, `"g_n"`：东西、北向分量。
- `"g_ee"`, `"g_nn"`, `"g_zz"`, `"g_en"`, `"g_ez"`, `"g_nz"`：重力梯度张量分量，单位 Eötvös。

### 复杂示例函数

- **`run_complex_example()`**：内置多棱柱示例（基底、矿化体、条带、凹陷等），带等值线与 `RdYlBu_r` 色标，结果写入 `forward_output/`。可直接在代码中调用，或通过 `python forword.py` 运行。

## 文件结构

```
forword/
├── README.md           # 本说明
├── requirements.txt   # 依赖
├── forword.py         # 正演与绘图
└── forward_output/    # 默认输出目录（运行后生成）
    ├── *.png          # 正演图像
    └── *.npy          # 正演结果数组
```

绘图时坐标轴与色标均以**普通数字**显示，不使用科学计数法。
