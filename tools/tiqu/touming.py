"""
将 xian-1.png 中的白色背景换成透明，保留非白色内容。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError as e:
    raise SystemExit("请先安装 Pillow: pip install Pillow") from e


def white_to_transparent(
    input_path: Path,
    output_path: Path,
    threshold: int = 240,
) -> None:
    """
    将接近白色的像素设为透明，其余保留。
    threshold: R、G、B 均大于该值视为白色背景（0-255）。
    """
    img = Image.open(input_path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    # np.asarray(PIL.Image) 可能返回只读视图，这里显式拷贝为可写数组
    arr = np.array(img, dtype=np.uint8, copy=True)
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]
    white_mask = (r > threshold) & (g > threshold) & (b > threshold)
    arr[white_mask, 3] = 0
    out = Image.fromarray(arr, mode="RGBA")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)
    transparent_count = int(white_mask.sum())
    print(f"已保存: {output_path}（透明化像素数: {transparent_count}，阈值 threshold={threshold}）")


if __name__ == "__main__":
    import argparse

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="将图片白色背景换成透明")
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "xian-1.png",
        help="输入图片路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "xian-1_transparent.png",
        help="输出 PNG（RGBA 透明底）",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=240,
        help="R/G/B 均大于该值视为白色背景（0-255，可调小以保留更多浅色）",
    )
    args = parser.parse_args()
    white_to_transparent(
        args.input,
        args.output,
        threshold=args.threshold,
    )
