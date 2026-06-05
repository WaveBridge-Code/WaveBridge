import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===============================
# 路径
# ===============================
GT_DIR   = "/path/to/classify/datasets/PET1_89_slice/test/"  # GT npy: [0,1]
PRED_DIR = "/path/to/2.5D_version1/logs/experiment/version_5/test/test_samples/pred_slices/"  # PRED npy: [-1,1]
OUT_DIR  = "/path/to/2.5D_version1/disease_select_subject/"  # CSV + 输出根目�?

# ===============================
# 可视化设置（彩色热力图）
# ===============================
dpi = 600
cmap = "turbo"
vmin, vmax = 0.0, 1.0

# 为了不和现有命名冲突，加前缀
PREFIX = "heat_"   # 输出文件名：heat_slice_XXX_GT.png / heat_slice_XXX_PRED.png

# ===============================
# 解析 slice index
# ===============================
def parse_slice_index(filename: str):
    m = re.search(r"_slice_(\d+)", filename)
    return int(m.group(1)) if m else None

# ===============================
# 归一�?
# ===============================
def normalize_gt(x: np.ndarray) -> np.ndarray:
    """GT: [0,1] -> [0,1]"""
    x = x.astype(np.float32)
    return np.clip(x, 0.0, 1.0)

def normalize_pred(x: np.ndarray) -> np.ndarray:
    """PRED: [-1,1] -> [0,1]"""
    x = x.astype(np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x + 1.0) / 2.0

# ===============================
# 保存单张切片（无边框高清�?
# ===============================
def save_heat_png(sl2d: np.ndarray, save_path: str):
    """
    sl2d: (H,W) in [0,1]
    """
    h, w = sl2d.shape

    fig = plt.figure(frameon=False)
    fig.set_size_inches(w / dpi, h / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(sl2d, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

# ===============================
# 处理一个切片：GT + PRED 各存一张彩色图
# ===============================
def process_one_slice(fn: str, out_subj_dir: str, slice_idx_fallback: int):
    gt_path = os.path.join(GT_DIR, fn)
    pr_path = os.path.join(PRED_DIR, fn)

    if not os.path.exists(gt_path):
        print(f"[WARN] Missing GT: {gt_path}")
        return
    if not os.path.exists(pr_path):
        print(f"[WARN] Missing PRED: {pr_path}")
        return

    gt = np.load(gt_path)
    pr = np.load(pr_path)

    # 如果�?(H,W,1) 这种，挤�?(H,W)
    if gt.ndim == 3 and gt.shape[-1] == 1:
        gt = gt[..., 0]
    if pr.ndim == 3 and pr.shape[-1] == 1:
        pr = pr[..., 0]

    if gt.ndim != 2 or pr.ndim != 2:
        raise ValueError(f"Expected 2D slices. Got GT {gt.shape}, PRED {pr.shape} for {fn}")

    # 归一化到 [0,1]
    gt = normalize_gt(gt)
    pr = normalize_pred(pr)

    # 可选：如果你之前做�?rot90 想保持一致，就取消注�?
    # gt = np.rot90(gt)
    # pr = np.rot90(pr)

    idx = parse_slice_index(fn)
    if idx is None:
        idx = slice_idx_fallback

    gt_save = os.path.join(out_subj_dir, f"{PREFIX}slice_{idx:03d}_GT.png")
    pr_save = os.path.join(out_subj_dir, f"{PREFIX}slice_{idx:03d}_PRED.png")

    save_heat_png(gt, gt_save)
    save_heat_png(pr, pr_save)

# ===============================
# 主流程：�?*_selected_slices.csv 找切片并导出
# ===============================
def main():
    csv_files = [f for f in os.listdir(OUT_DIR) if f.endswith("_selected_slices.csv")]
    if len(csv_files) == 0:
        print(f"ERROR: No *_selected_slices.csv found in {OUT_DIR}")
        return

    print("Found selected csv files:", len(csv_files))

    for csv_name in csv_files:
        csv_path = os.path.join(OUT_DIR, csv_name)
        sdf = pd.read_csv(csv_path)

        if "subject_id" not in sdf.columns or "filename" not in sdf.columns:
            print(f"[WARN] Skip {csv_name}: missing columns (subject_id, filename)")
            continue

        # 每个 subject 建文件夹：OUT_DIR/<ID>/
        for sid, g in sdf.groupby("subject_id"):
            out_subj_dir = os.path.join(OUT_DIR, sid)
            os.makedirs(out_subj_dir, exist_ok=True)

            for i, row in g.reset_index(drop=True).iterrows():
                fn = str(row["filename"]).strip()
                process_one_slice(fn, out_subj_dir, slice_idx_fallback=i)

        print(f"Done: {csv_name}")

    print("All done. Heat PNGs saved under:", OUT_DIR)

if __name__ == "__main__":
    main()