import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================================================
# 路径设置
# =========================================================
PRED_DIR = "/path/to/2.5D_128/logs/experiment/version_4/test/test_samples/pred_overlap_avg_nii/"
GT_DIR   = "/path/to/classify/datasets/PET_99_test/"

LABEL_XLSX = "/path/to/classify/datasets/MRI_GT_PET_IDs.xlsx"

OUT_DIR = "/path/to/2.5D_128/logs/experiment/version_4/test/test_samples/ssp_like_zscore/"


# =========================================================
# 参数设置
# =========================================================
EPS = 1e-6

# 只用 CN 作为正常数据�?
NORMAL_LABEL = "CN"

# PET 背景阈�?
BRAIN_MASK_THR = 0.01

# 3D-SSP / Z-score 推荐开�?
DO_GLOBAL_MEAN_NORM = True

# Z-score 可视化范�?
Z_VMIN = 0.0
Z_VMAX = 5.0
Z_CMAP = "turbo"

SAVE_Z_NII = True
DPI = 300


# =========================================================
# 工具函数
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def strip_nii_suffix(path):
    name = os.path.basename(path)
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return name


def list_nii_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.nii.gz")))
    files += sorted(glob.glob(os.path.join(folder, "*.nii")))
    return files


def to_01_by_range(x, name=""):
    """
    判断 PET 数值范围：
    1. 如果�?[-1,1]，转�?[0,1]
    2. 如果�?[0,1]，保持不�?
    3. 其他范围，做 min-max normalize
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    xmin = float(np.min(x))
    xmax = float(np.max(x))

    if xmin < -0.05 and xmin >= -1.2 and xmax <= 1.2:
        print(f"[Range] {name}: [{xmin:.4f}, {xmax:.4f}] -> [-1,1] to [0,1]")
        x = np.clip(x, -1.0, 1.0)
        x = (x + 1.0) / 2.0
        return x

    if xmin >= -0.05 and xmax <= 1.2:
        print(f"[Range] {name}: [{xmin:.4f}, {xmax:.4f}] -> [0,1], keep")
        x = np.clip(x, 0.0, 1.0)
        return x

    print(f"[Range] {name}: [{xmin:.4f}, {xmax:.4f}] -> unusual, min-max normalize")
    x = (x - xmin) / (xmax - xmin + EPS)
    x = np.clip(x, 0.0, 1.0)
    return x


def make_brain_mask(x):
    return x > BRAIN_MASK_THR


def global_mean_normalize(x, mask):
    """
    global normalization:
    用脑区内平均值归一化�?
    """
    if not DO_GLOBAL_MEAN_NORM:
        return x

    vals = x[mask]
    if vals.size == 0:
        return x

    gmean = float(np.mean(vals))
    if gmean < EPS:
        return x

    return x / (gmean + EPS)


def load_pet_canonical(path):
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    data = np.squeeze(data)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape={data.shape}, path={path}")

    return data, img.affine, img.header


def load_cn_gt_files(gt_dir, label_xlsx):
    """
    �?MRI_GT_PET_IDs.xlsx 中读�?ID �?label�?
    只选择 label == CN �?GT PET 作为 normal database�?
    """
    df = pd.read_excel(label_xlsx)

    if df.shape[1] < 2:
        raise ValueError("Excel 至少需要两列：第一�?ID，第二列 label")

    id_col = df.columns[0]
    label_col = df.columns[1]

    print("=" * 80)
    print(f"Label file: {label_xlsx}")
    print(f"Detected ID column   : {id_col}")
    print(f"Detected label column: {label_col}")
    print("=" * 80)

    df[id_col] = df[id_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()

    cn_ids = set(df.loc[df[label_col] == NORMAL_LABEL, id_col].tolist())

    print(f"Total IDs in Excel: {len(df)}")
    print(f"CN IDs for normal database: {len(cn_ids)}")

    all_gt_files = list_nii_files(gt_dir)
    gt_map = {strip_nii_suffix(p): p for p in all_gt_files}

    cn_gt_files = []
    missing_ids = []

    for sid in sorted(cn_ids):
        if sid in gt_map:
            cn_gt_files.append(gt_map[sid])
        else:
            missing_ids.append(sid)

    print(f"GT PET files in folder: {len(all_gt_files)}")
    print(f"Matched CN GT PET files: {len(cn_gt_files)}")
    print(f"Missing CN GT PET files: {len(missing_ids)}")

    if len(missing_ids) > 0:
        print("\n[WARN] Some CN IDs in Excel were not found in GT_DIR. First 20 missing IDs:")
        for sid in missing_ids[:20]:
            print("  ", sid)

    if len(cn_gt_files) == 0:
        raise RuntimeError("No CN GT PET files found. Please check ID names and GT_DIR.")

    return cn_gt_files, df, id_col, label_col


def build_normal_database(gt_files):
    """
    �?CN GT PET 构建 normal mean / std�?
    """
    normed_list = []
    ref_shape = None
    ref_affine = None
    ref_header = None

    print("\nBuilding CN normal database...")
    print(f"CN GT files used: {len(gt_files)}")

    for path in tqdm(gt_files):
        sid = strip_nii_suffix(path)

        x, affine, header = load_pet_canonical(path)
        x = to_01_by_range(x, name=f"CN_GT/{sid}")

        mask = make_brain_mask(x)
        x = global_mean_normalize(x, mask)

        if ref_shape is None:
            ref_shape = x.shape
            ref_affine = affine
            ref_header = header
        else:
            if x.shape != ref_shape:
                raise ValueError(
                    f"Shape mismatch in CN GT database:\n"
                    f"ref_shape={ref_shape}\n"
                    f"current_shape={x.shape}\n"
                    f"path={path}\n"
                    f"Please make sure all GT PET are aligned and have the same shape."
                )

        normed_list.append(x.astype(np.float32))

    arr = np.stack(normed_list, axis=0)

    normal_mean = np.mean(arr, axis=0)
    normal_std = np.std(arr, axis=0)

    # 防止 std 太小导致 Z-score 爆炸
    normal_std = np.maximum(normal_std, 0.05)

    normal_mask = normal_mean > BRAIN_MASK_THR

    return normal_mean, normal_std, normal_mask, ref_affine, ref_header


def compute_negative_zscore(x_norm, normal_mean, normal_std, normal_mask):
    """
    negative Z-score:
    normal_mean - subject_value 越大，表示该区域代谢越低�?
    """
    z = (normal_mean - x_norm) / (normal_std + EPS)

    # 只保留低代谢方向
    z = np.maximum(z, 0.0)

    # 背景�?0
    z[~normal_mask] = 0.0

    return z.astype(np.float32)


def right_lateral_projection(z):
    """
    简化版 right hemisphere lateral projection�?
    注意：这不是严格�?Neurostat 3D-SSP，只是近似右半球侧位投影�?

    nib.as_closest_canonical 后通常�?RAS�?
    axis 0: Left-Right
    axis 1: Posterior-Anterior
    axis 2: Inferior-Superior
    """
    nx, ny, nz = z.shape
    mid = nx // 2

    right_hemi = z[mid:, :, :]

    # 沿左右方向做最大投�?
    proj = np.max(right_hemi, axis=0)

    # 调整显示方向
    proj = proj.T
    proj = np.flipud(proj)

    return proj


def save_projection_png(proj, out_path, title=None):
    plt.figure(figsize=(4, 3), dpi=DPI)
    plt.imshow(
        proj,
        cmap=Z_CMAP,
        vmin=Z_VMIN,
        vmax=Z_VMAX,
        interpolation="nearest"
    )
    plt.axis("off")

    if title is not None:
        plt.title(title, fontsize=8)

    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_z_nii(z, affine, header, out_path):
    img = nib.Nifti1Image(z.astype(np.float32), affine=affine, header=header)
    nib.save(img, out_path)


# =========================================================
# 主程�?
# =========================================================
def main():
    ensure_dir(OUT_DIR)

    znii_dir = os.path.join(OUT_DIR, "zscore_nii")
    png_dir = os.path.join(OUT_DIR, "right_lateral_png")
    ensure_dir(znii_dir)
    ensure_dir(png_dir)

    cn_gt_files, df_label, id_col, label_col = load_cn_gt_files(
        gt_dir=GT_DIR,
        label_xlsx=LABEL_XLSX
    )

    pred_files = list_nii_files(PRED_DIR)

    if len(pred_files) == 0:
        raise RuntimeError(f"No prediction nii.gz found in {PRED_DIR}")

    print("=" * 80)
    print(f"PRED_DIR: {PRED_DIR}")
    print(f"GT_DIR  : {GT_DIR}")
    print(f"OUT_DIR : {OUT_DIR}")
    print(f"Number of PRED files: {len(pred_files)}")
    print("=" * 80)

    # 保存本次使用�?CN normal database 列表，方便之后论�?实验记录
    used_cn_csv = os.path.join(OUT_DIR, "used_CN_normal_database.csv")
    used_cn_df = pd.DataFrame({
        "ID": [strip_nii_suffix(p) for p in cn_gt_files],
        "GT_path": cn_gt_files
    })
    used_cn_df.to_csv(used_cn_csv, index=False)
    print(f"Used CN normal database list saved to: {used_cn_csv}")

    normal_mean, normal_std, normal_mask, ref_affine, ref_header = build_normal_database(cn_gt_files)

    # 保存 normal database
    nib.save(
        nib.Nifti1Image(normal_mean.astype(np.float32), ref_affine, ref_header),
        os.path.join(OUT_DIR, "CN_normal_mean.nii.gz")
    )
    nib.save(
        nib.Nifti1Image(normal_std.astype(np.float32), ref_affine, ref_header),
        os.path.join(OUT_DIR, "CN_normal_std.nii.gz")
    )

    print("\nProcessing prediction PET...")

    for pred_path in tqdm(pred_files):
        sid = strip_nii_suffix(pred_path)

        x, affine, header = load_pet_canonical(pred_path)
        x = to_01_by_range(x, name=f"PRED/{sid}")

        if x.shape != normal_mean.shape:
            raise ValueError(
                f"Shape mismatch between prediction and CN normal database:\n"
                f"prediction shape={x.shape}\n"
                f"normal shape={normal_mean.shape}\n"
                f"path={pred_path}\n"
                f"Please make sure prediction PET and GT PET are in the same space and size."
            )

        mask = make_brain_mask(x)
        x_norm = global_mean_normalize(x, mask)

        z = compute_negative_zscore(
            x_norm=x_norm,
            normal_mean=normal_mean,
            normal_std=normal_std,
            normal_mask=normal_mask
        )

        if SAVE_Z_NII:
            z_path = os.path.join(znii_dir, f"{sid}_neg_zscore.nii.gz")
            save_z_nii(z, affine, header, z_path)

        proj = right_lateral_projection(z)
        png_path = os.path.join(png_dir, f"{sid}_right_lateral.png")
        save_projection_png(proj, png_path)

    print("\nDone.")
    print(f"CN normal mean saved to: {os.path.join(OUT_DIR, 'CN_normal_mean.nii.gz')}")
    print(f"CN normal std saved to : {os.path.join(OUT_DIR, 'CN_normal_std.nii.gz')}")
    print(f"Z-score NIfTI saved to : {znii_dir}")
    print(f"Right lateral PNG saved to: {png_dir}")


if __name__ == "__main__":
    main()