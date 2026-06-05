import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path


def convert_npy_to_images(input_root, output_root):
    """
    �?3D .npy 文件（Z,H,W �?1,1,H,W 等）逐层转换为图片�?
    自动检测是否需要反归一化（[-1,1] �?[0,1]），无控制台刷屏输出�?
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    npy_files = list(input_root.rglob("*.npy"))
    warned_once = False  # 控制只提示一�?

    for npy_path in tqdm(npy_files, desc="Converting .npy to images"):
        data = np.load(npy_path)
        data = np.squeeze(data)  # 去掉多余维度

        # 检查是否需要反归一化（静默处理�?
        if data.min() < 0:
            if not warned_once:
                print("⚠️ 检测到部分数据�?[-1,1] 范围，将自动反归一化为 [0,1]�?)
                warned_once = True
            data = (data + 1) / 2

        data = np.clip(data, 0, 1)

        # 处理三维�?
        if data.ndim == 3:
            for i in range(data.shape[0]):
                img = (data[i] * 255).astype(np.uint8)
                img = Image.fromarray(img)

                relative_path = npy_path.relative_to(input_root)
                subfolder = output_root / relative_path.stem
                subfolder.mkdir(parents=True, exist_ok=True)

                output_img_path = subfolder / f"slice_{i:03d}.png"
                img.save(output_img_path)

        elif data.ndim == 2:
            img = (data * 255).astype(np.uint8)
            img = Image.fromarray(img)

            relative_path = npy_path.relative_to(input_root)
            output_img_path = output_root / relative_path.with_suffix(".png")
            output_img_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(output_img_path)

    print("�?所有切片图片已保存完成�?)


def main():
    input_root = "/path/to/SelfRDB_skip_hf/logs/experiment/version_21/test/test_samples/pred_slices"
    output_root = "/path/to/SelfRDB_skip_hf/logs/experiment/version_21/test/test_samples/pred_slices_image"
    convert_npy_to_images(input_root, output_root)


if __name__ == "__main__":
    main()
