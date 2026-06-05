import os
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm


# =========================
# Path settings
# =========================
INPUT_DIR = "/path/to/2.5D_128/logs/experiment/version_4/test/test_samples/pred_overlap_avg_nii"
OUTPUT_DIR = "/path/to/2.5D_128/logs/experiment/version_4/test/test_samples/norm_pred"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# Tolerance settings
# =========================
# 允许预测结果轻微超过 [-1, 1]
# 例如你的结果�?[-1.051, 1.024]，这种属于轻微越�?
MINUS_ONE_ONE_LOWER_TOL = -1.20
MINUS_ONE_ONE_UPPER_TOL = 1.20

# 判断是否已经�?[0, 1] 的容忍范�?
ZERO_ONE_LOWER_TOL = -0.05
ZERO_ONE_UPPER_TOL = 1.05

# 统计极端值时使用的分位数，更稳定
LOW_PERCENTILE = 0.1
HIGH_PERCENTILE = 99.9


def collect_nii_files(input_dir):
    files = []

    for name in sorted(os.listdir(input_dir)):
        if name.endswith(".nii.gz"):
            files.append(os.path.join(input_dir, name))

    return files


def load_nii_data(nii_path):
    img = nib.load(nii_path)
    data = img.get_fdata(dtype=np.float32)

    return img, data


def get_range_info(nii_path):
    img, data = load_nii_data(nii_path)

    finite_mask = np.isfinite(data)
    if not np.any(finite_mask):
        raise ValueError(f"No finite values found in: {nii_path}")

    valid = data[finite_mask]

    info = {
        "filename": os.path.basename(nii_path),
        "shape": str(data.shape),
        "min": float(valid.min()),
        "p0_1": float(np.percentile(valid, LOW_PERCENTILE)),
        "p1": float(np.percentile(valid, 1)),
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "p99": float(np.percentile(valid, 99)),
        "p99_9": float(np.percentile(valid, HIGH_PERCENTILE)),
        "max": float(valid.max()),
        "nan_count": int(np.isnan(data).sum()),
        "inf_count": int(np.isinf(data).sum()),
    }

    return info


def decide_normalization_mode(global_min, global_max, global_p0_1, global_p99_9):
    """
    返回:
        minus1_1_to_0_1_clip:
            数据主体接近 [-1, 1]，允许轻微越界�?
            �?clip �?[-1, 1]，再映射�?[0, 1]�?

        already_0_1_clip:
            数据主体接近 [0, 1]，允许轻微越界�?
            直接 clip �?[0, 1]�?

        minmax_per_volume:
            极端备用方案，不推荐默认使用�?
    """

    # 情况1：整体接�?[-1, 1]，包括轻微越�?
    if global_min >= MINUS_ONE_ONE_LOWER_TOL and global_max <= MINUS_ONE_ONE_UPPER_TOL:
        if global_p0_1 < 0:
            return "minus1_1_to_0_1_clip"

    # 情况2：整体接�?[0, 1]，包括轻微越�?
    if global_min >= ZERO_ONE_LOWER_TOL and global_max <= ZERO_ONE_UPPER_TOL:
        return "already_0_1_clip"

    # 情况3：如果极少数异常值导致范围很大，但主体仍接近 [-1, 1]
    if global_p0_1 >= -1.10 and global_p99_9 <= 1.10 and global_p0_1 < 0:
        return "minus1_1_to_0_1_clip"

    # 情况4：如果极少数异常值导致范围很大，但主体仍接近 [0, 1]
    if global_p0_1 >= -0.02 and global_p99_9 <= 1.02:
        return "already_0_1_clip"

    raise ValueError(
        "Cannot safely decide normalization mode.\n"
        f"global_min={global_min:.6f}, global_max={global_max:.6f}, "
        f"global_p0.1={global_p0_1:.6f}, global_p99.9={global_p99_9:.6f}\n"
        "Please check the intensity distribution manually."
    )


def normalize_data(data, mode):
    data = data.astype(np.float32)

    # 先处�?NaN / Inf
    data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)

    if mode == "minus1_1_to_0_1_clip":
        # 稳定处理轻微越界预测�?
        data = np.clip(data, -1.0, 1.0)
        data = (data + 1.0) / 2.0
        data = np.clip(data, 0.0, 1.0)
        return data.astype(np.float32)

    elif mode == "already_0_1_clip":
        # 已经�?[0, 1]，只处理轻微越界
        data = np.clip(data, 0.0, 1.0)
        return data.astype(np.float32)

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def save_nifti_like(ref_img, data, out_path):
    new_header = ref_img.header.copy()
    new_header.set_data_dtype(np.float32)

    new_img = nib.Nifti1Image(
        data.astype(np.float32),
        affine=ref_img.affine,
        header=new_header
    )

    nib.save(new_img, out_path)


def main():
    nii_files = collect_nii_files(INPUT_DIR)

    if len(nii_files) == 0:
        raise FileNotFoundError(f"No .nii.gz files found in: {INPUT_DIR}")

    print("=" * 80)
    print(f"Input dir : {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Found {len(nii_files)} .nii.gz files")
    print("=" * 80)

    # =========================
    # Step 1: check range before normalization
    # =========================
    before_records = []

    print("\nChecking value ranges before normalization...")
    for nii_path in tqdm(nii_files):
        info = get_range_info(nii_path)
        before_records.append(info)

    before_df = pd.DataFrame(before_records)

    before_csv = os.path.join(OUTPUT_DIR, "before_normalization_range.csv")
    before_df.to_csv(before_csv, index=False)

    global_min = before_df["min"].min()
    global_max = before_df["max"].max()
    global_p0_1 = before_df["p0_1"].min()
    global_p99_9 = before_df["p99_9"].max()

    print("\n" + "=" * 80)
    print("Global range before normalization")
    print(f"Global min   : {global_min:.6f}")
    print(f"Global p0.1  : {global_p0_1:.6f}")
    print(f"Global p99.9 : {global_p99_9:.6f}")
    print(f"Global max   : {global_max:.6f}")
    print("=" * 80)
    print(f"Saved before-normalization report to: {before_csv}")

    # =========================
    # Step 2: decide normalization mode
    # =========================
    mode = decide_normalization_mode(
        global_min=global_min,
        global_max=global_max,
        global_p0_1=global_p0_1,
        global_p99_9=global_p99_9,
    )

    print("\n" + "=" * 80)
    print(f"Normalization mode: {mode}")

    if mode == "minus1_1_to_0_1_clip":
        print("Detected predicted PET is approximately in [-1, 1].")
        print("Processing: clip to [-1, 1], then map to [0, 1].")
        print("Formula: x_norm = (clip(x, -1, 1) + 1) / 2")

    elif mode == "already_0_1_clip":
        print("Detected predicted PET is approximately in [0, 1].")
        print("Processing: clip to [0, 1], no extra normalization.")

    print("=" * 80)

    # =========================
    # Step 3: normalize and save
    # =========================
    after_records = []

    print("\nSaving normalized NIfTI files...")
    for nii_path in tqdm(nii_files):
        img, data = load_nii_data(nii_path)

        norm_data = normalize_data(data, mode)

        out_path = os.path.join(OUTPUT_DIR, os.path.basename(nii_path))
        save_nifti_like(img, norm_data, out_path)

        finite = norm_data[np.isfinite(norm_data)]

        after_records.append({
            "filename": os.path.basename(nii_path),
            "shape": str(norm_data.shape),
            "min": float(finite.min()),
            "p0_1": float(np.percentile(finite, LOW_PERCENTILE)),
            "p1": float(np.percentile(finite, 1)),
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "p99": float(np.percentile(finite, 99)),
            "p99_9": float(np.percentile(finite, HIGH_PERCENTILE)),
            "max": float(finite.max()),
            "saved_path": out_path,
        })

    after_df = pd.DataFrame(after_records)
    after_csv = os.path.join(OUTPUT_DIR, "after_normalization_range.csv")
    after_df.to_csv(after_csv, index=False)

    print("\n" + "=" * 80)
    print("Finished normalization.")
    print(f"Saved normalized files to: {OUTPUT_DIR}")
    print(f"Saved after-normalization report to: {after_csv}")
    print("=" * 80)

    print("\nGlobal range after normalization")
    print(f"Global min   : {after_df['min'].min():.6f}")
    print(f"Global p0.1  : {after_df['p0_1'].min():.6f}")
    print(f"Global p99.9 : {after_df['p99_9'].max():.6f}")
    print(f"Global max   : {after_df['max'].max():.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()