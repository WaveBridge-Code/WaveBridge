import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# ===============================
# 路径
# ===============================
GT_DIR = "/path/to/classify/datasets/PET_slice_99/test/"
PRED_DIR = "/path/to/2.5D_128/logs/experiment/version_4/test/test_samples/pred_slices/"
OUT_DIR = "/path/to/2.5D_128/disease_metric/"
TEST_CSV = "/path/to/2.5D_version1/test_labels.csv"

os.makedirs(OUT_DIR, exist_ok=True)


# ===============================
# 解析 subject_id / slice_idx
# ===============================
def parse_subject_id(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    return base.split("_slice_")[0]


def parse_slice_index(filename: str):
    m = re.search(r"_slice_(\d+)", filename)
    return int(m.group(1)) if m else None


# ===============================
# 数据归一�?
# ===============================
def norm_gt(x):
    """
    GT PET 默认已经�?[0, 1]
    """
    x = np.asarray(x, dtype=np.float32)
    return np.clip(x, 0.0, 1.0)


def norm_pred(x):
    """
    预测 PET 默认保存�?[-1, 1]
    转换�?[0, 1]
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x + 1.0) / 2.0


# ===============================
# 计算单张切片指标
# ===============================
def compute_metrics(gt, pr):
    if gt.shape != pr.shape:
        return None

    gt = norm_gt(gt)
    pr = norm_pred(pr)

    mae = float(np.mean(np.abs(gt - pr)))
    ps = float(psnr(gt, pr, data_range=1.0))

    if gt.ndim == 2:
        ss = float(ssim(gt, pr, data_range=1.0))
    else:
        ss = float(ssim(gt, pr, data_range=1.0, channel_axis=-1))

    return ss, ps, mae


# ===============================
# 读取 test.csv：ID -> label
# ===============================
def load_id_to_label_map(csv_path: str):
    df = pd.read_csv(csv_path)

    id_candidates = ["ID"]
    label_candidates = ["Type"]

    id_col = next((c for c in id_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if id_col is None or label_col is None:
        raise ValueError(
            f"Cannot find id/label columns in {csv_path}. "
            f"Columns are: {list(df.columns)}."
        )

    df[id_col] = df[id_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip().str.upper()

    valid_labels = {"AD", "CN", "MCI"}
    df = df[df[label_col].isin(valid_labels)].copy()

    id2label = dict(zip(df[id_col], df[label_col]))

    return id2label, id_col, label_col, df


# ===============================
# 主函�?
# ===============================
def main():
    # ---------------------------------------------------
    # 1. 读取 ID 与类别映�?
    # ---------------------------------------------------
    id2label, id_col, label_col, test_df = load_id_to_label_map(TEST_CSV)
    print(f"[test.csv] using id_col='{id_col}', label_col='{label_col}', n={len(test_df)}")

    # ---------------------------------------------------
    # 2. 读取 GT / Pred 文件
    # ---------------------------------------------------
    gt_list = [f for f in os.listdir(GT_DIR) if f.endswith(".npy")]
    pred_list = [f for f in os.listdir(PRED_DIR) if f.endswith(".npy")]

    gt_files = {f: os.path.join(GT_DIR, f) for f in gt_list}
    pred_files = {f: os.path.join(PRED_DIR, f) for f in pred_list}

    print("GT files:", len(gt_files))
    print("PRED files:", len(pred_files))

    common = sorted(set(gt_files.keys()) & set(pred_files.keys()))
    print("Common filenames:", len(common))

    if len(common) == 0:
        print("\nERROR: No matched GT/PRED filenames found.")
        print("Example GT names:", list(gt_files.keys())[:5])
        print("Example PRED names:", list(pred_files.keys())[:5])
        return

    # ---------------------------------------------------
    # 3. �?slice 计算指标
    # ---------------------------------------------------
    rows = []

    for fn in tqdm(common, desc="Computing slice metrics"):
        sid = parse_subject_id(fn)
        slice_idx = parse_slice_index(fn)

        if sid not in id2label:
            continue

        if slice_idx is None:
            continue

        gt = np.load(gt_files[fn]).astype(np.float32)
        pr = np.load(pred_files[fn]).astype(np.float32)

        result = compute_metrics(gt, pr)
        if result is None:
            continue

        ss, ps, mae = result

        rows.append({
            "subject_id": sid,
            "label": id2label[sid],
            "filename": fn,
            "slice_idx": slice_idx,
            "ssim": ss,
            "psnr": ps,
            "mae": mae
        })

    slice_df = pd.DataFrame(rows)

    if slice_df.empty:
        print("ERROR: No valid slice metrics were computed.")
        return

    # ---------------------------------------------------
    # 4. 按病例聚合所有切片指�?
    # ---------------------------------------------------
    subject_df = slice_df.groupby(["subject_id", "label"]).agg(
        n_slices=("ssim", "count"),
        ssim_mean=("ssim", "mean"),
        ssim_median=("ssim", "median"),
        ssim_std=("ssim", "std"),
        psnr_mean=("psnr", "mean"),
        psnr_median=("psnr", "median"),
        psnr_std=("psnr", "std"),
        mae_mean=("mae", "mean"),
        mae_median=("mae", "median"),
        mae_std=("mae", "std"),
    ).reset_index()

    # ---------------------------------------------------
    # 5. 分别保存 AD / CN / MCI 所有病例指�?
    # ---------------------------------------------------
    class_order = ["AD", "CN", "MCI"]

    for cls in class_order:
        cls_df = subject_df[subject_df["label"] == cls].copy()

        # 默认�?SSIM 中位数从高到低排序，便于查看
        cls_df = cls_df.sort_values(
            by="ssim_median",
            ascending=False
        ).reset_index(drop=True)

        save_path = os.path.join(OUT_DIR, f"{cls}_metrics.csv")
        cls_df.to_csv(save_path, index=False)

        print(f"[Saved] {cls}: {len(cls_df)} subjects -> {save_path}")

    print("\nFinished.")
    print("All class-wise subject metrics have been saved in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()