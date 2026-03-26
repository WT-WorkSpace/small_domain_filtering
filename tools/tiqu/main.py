"""
从 xian.png 中提取近黑色像素（断裂线等），保存为 PNG：线段不透明，其余透明。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("请先安装 Pillow: pip install Pillow") from e


def extract_black_lines(
    input_path: Path,
    output_path: Path,
    threshold: int = 55,
    min_rgb_sum: Optional[int] = None,
) -> None:
    """
    threshold: R、G、B 均小于该值视为黑色线段（避免深蓝底被误提，因蓝通道往往较高）。
    min_rgb_sum: 若给定，需同时满足 R+G+B < min_rgb_sum（更严）；默认仅用三通道均 < threshold。
    """
    img = Image.open(input_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    mask = (r < threshold) & (g < threshold) & (b < threshold)
    if min_rgb_sum is not None:
        mask = mask & (r.astype(np.int32) + g + b < min_rgb_sum)

    h, w = arr.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask, 0] = 0
    rgba[mask, 1] = 0
    rgba[mask, 2] = 0
    rgba[mask, 3] = 255
    rgba[~mask, 3] = 0

    out = Image.fromarray(rgba, mode="RGBA")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    n = int(mask.sum())
    print(f"已保存: {output_path}（提取像素数: {n}，阈值 threshold={threshold}）")


if __name__ == "__main__":
    import argparse

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="提取图中黑色线段，背景透明")
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "xian.png",
        help="输入图片路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "xian_lines_rgba.png",
        help="输出 PNG（RGBA 透明底）",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=120,
        help="R/G/B 均小于该值视为黑线（0-255，可调大提更多灰线，调小更纯黑）",
    )
    parser.add_argument(
        "--rgb-sum-max",
        type=int,
        default=None,
        help="可选：R+G+B 须小于该值，与 threshold 同时生效，用于压制深蓝噪声",
    )
    args = parser.parse_args()
    extract_black_lines(
        args.input,
        args.output,
        threshold=args.threshold,
        min_rgb_sum=args.rgb_sum_max,
    )
