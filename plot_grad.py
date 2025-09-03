import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取数据
data = pd.read_excel("data2.xlsx", header=None).values
print(f"数据维度: {data.shape[0]} x {data.shape[1]}")

# 2. 计算梯度
gy, gx = np.gradient(data)   # 注意：numpy 的顺序是 (行方向, 列方向)
magnitude = np.sqrt(gx**2 + gy**2)
u = gx / (magnitude + 1e-10)
v = gy / (magnitude + 1e-10)

# === 新增：计算梯度方向（弧度 & 角度） ===
theta_rad = np.arctan2(gy, gx)        # [-pi, pi]
theta_deg = np.degrees(theta_rad)     # 转换为角度 [-180, 180]

print("部分梯度方向（度）：")
print(theta_deg[:5, :5])   # 打印前 5x5 个方向，避免太大

# 3. 创建坐标
x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

# 4. 绘制梯度方向图
plt.figure(figsize=(8, 6))
plt.quiver(x, y, u, v, color='r', scale=20)
plt.title("梯度方向场")
plt.xlabel("X坐标")
plt.ylabel("Y坐标")
plt.axis("equal")
plt.grid(True)
plt.show()

print("梯度方向图绘制完成！")

