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
OUT_DIR = "/path/to/2.5D_128/disease_select_subject/"
TEST_CSV = "/path/to/2.5D_version1/test_labels.csv"
os.makedirs(OUT_DIR, exist_ok=True)

# ===============================
# 仍然�?subject �?Top3
# ===============================
TOP_PER_CLASS = 5

# ===============================
# 只保留关键序�?[30, 70]
# ===============================
SLICE_START = 30
SLICE_END = 70

# ===============================
# subject-level 排序指标
# 可�?
# "ssim_mean" / "ssim_median" / "psnr_mean" / "mae_mean"
# 注意 mae 越小越好
# ===============================
RANK_BY = "ssim_median"

# ===============================
# 关键序列内部�?slice 排序指标
# 可�? "ssim" / "psnr" / "mae"
# ===============================
SLICE_RANK_BY = "ssim"

# ===============================
# 解析 subject_id / slice_idx
# ===============================
def parse_subject_id(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    return base.split("_slice_")[0]

def parse_slice_index(filename: str):
    m = re.search(r"_slice_(\d+)", filename)
    return int(m.group(1)) if m else None

def norm_gt(x):
    x = np.asarray(x, dtype=np.float32)
    return np.clip(x, 0.0, 1.0)

def norm_pred(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    return (x + 1.0) / 2.0

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
# 读取 test.csv：自动猜 ID�?/ 类别�?
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

    valid = {"AD", "MCI", "CN"}
    df = df[df[label_col].isin(valid)].copy()

    id2label = dict(zip(df[id_col], df[label_col]))
    return id2label, id_col, label_col, df

def main():
    # 读取类别映射
    id2label, id_col, label_col, test_df = load_id_to_label_map(TEST_CSV)
    print(f"[test.csv] using id_col='{id_col}', label_col='{label_col}', n={len(test_df)}")

    # 读取文件�?
    gt_list = [f for f in os.listdir(GT_DIR) if f.endswith(".npy")]
    pred_list = [f for f in os.listdir(PRED_DIR) if f.endswith(".npy")]

    gt_files = {f: os.path.join(GT_DIR, f) for f in gt_list}
    pred_files = {f: os.path.join(PRED_DIR, f) for f in pred_list}

    print("GT files:", len(gt_files))
    print("PRED files:", len(pred_files))

    common = sorted(set(gt_files.keys()) & set(pred_files.keys()))
    print("Common filenames:", len(common))

    if len(common) == 0:
        print("\nExample GT names:", list(gt_files.keys())[:5])
        print("\nExample PRED names:", list(pred_files.keys())[:5])
        return

    # ===============================
    # �?slice 计算指标
    # ===============================
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

    df = pd.DataFrame(rows)
    if df.empty:
        print("ERROR: df is empty. No valid matched pairs after label filtering.")
        df.to_csv(os.path.join(OUT_DIR, "slice_metrics_EMPTY.csv"), index=False)
        return

    # 保存全部 slice 指标
    df.to_csv(os.path.join(OUT_DIR, "slice_metrics.csv"), index=False)

    # ===============================
    # �?subject 聚合（全切片上做统计�?
    # ===============================
    agg = df.groupby(["subject_id", "label"]).agg(
        n_slices=("ssim", "count"),
        ssim_mean=("ssim", "mean"),
        ssim_median=("ssim", "median"),
        psnr_mean=("psnr", "mean"),
        mae_mean=("mae", "mean"),
    ).reset_index()

    ascending = (RANK_BY == "mae_mean")
    agg = agg.sort_values(RANK_BY, ascending=ascending)
    agg.to_csv(os.path.join(OUT_DIR, "subject_metrics_with_label.csv"), index=False)

    # ===============================
    # 每个类别�?Top3 subject
    # ===============================
    selected_all = []
    for cls in ["AD", "MCI", "CN"]:
        sub = agg[agg["label"] == cls].copy()
        if sub.empty:
            print(f"[WARN] No subjects found for class {cls}")
            continue

        chosen = sub.head(TOP_PER_CLASS).copy()
        chosen.to_csv(os.path.join(OUT_DIR, f"top_{TOP_PER_CLASS}_{cls}.csv"), index=False)
        selected_all.append(chosen)

        print(f"{cls} selected subject_id:", list(chosen["subject_id"]))

    if len(selected_all) == 0:
        print("ERROR: No class-wise subjects selected.")
        return

    selected_df = pd.concat(selected_all, ignore_index=True)
    selected_df.to_csv(
        os.path.join(OUT_DIR, f"selected_top_{TOP_PER_CLASS}_per_class.csv"),
        index=False
    )

    # ===============================
    # 对入选的 Top3 subject
    # 只保留关键序�?[30, 70]
    # 并导出：
    # 1) �?slice 顺序
    # 2) 按质量排�?
    # ===============================
    for _, r in selected_df.iterrows():
        sid = r["subject_id"]
        cls = r["label"]

        sdf = df[df["subject_id"] == sid].copy()
        sdf = sdf[(sdf["slice_idx"] >= SLICE_START) & (sdf["slice_idx"] <= SLICE_END)].copy()

        if sdf.empty:
            print(f"[WARN] No slices in [{SLICE_START}, {SLICE_END}] for {cls}_{sid}")
            continue

        # 1) 按切片顺序保�?
        sdf_in_order = sdf.sort_values("slice_idx").copy()
        sdf_in_order.to_csv(
            os.path.join(OUT_DIR, f"{cls}_{sid}_selected_slices.csv"),
            index=False
        )

        # 2) 按关键序列内部质量排序保�?
        if SLICE_RANK_BY not in ["ssim", "psnr", "mae"]:
            raise ValueError(f"Unsupported SLICE_RANK_BY={SLICE_RANK_BY}")

        slice_ascending = (SLICE_RANK_BY == "mae")
        sdf_ranked = sdf.sort_values(SLICE_RANK_BY, ascending=slice_ascending).copy()
        sdf_ranked.to_csv(
            os.path.join(OUT_DIR, f"{cls}_{sid}_selected_slices_ranked_by_{SLICE_RANK_BY}.csv"),
            index=False
        )

    print("Finished. Results saved in:", OUT_DIR)

if __name__ == "__main__":
    main()