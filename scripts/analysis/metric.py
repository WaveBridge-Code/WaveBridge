import os
import pandas as pd
import numpy as np

# =========================
# 配置
# =========================
EXCEL_PATH = "/path/to/clicinal/result/GT_PET.xlsx"
CLASSES = ["CN", "MCI", "AD"]


def normalize_label(x):
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    if x in ["NC", "NORMAL"]:
        return "CN"
    return x


def read_excel_with_auto_header(path):
    """自动寻找包含 class �?Diagnosis label 的表头行"""
    raw = pd.read_excel(path, sheet_name=0, header=None)

    header_row = None
    for i in range(len(raw)):
        row_values = [str(v).strip() for v in raw.iloc[i].tolist()]
        row_text = " ".join(row_values).lower()

        if "class" in row_text and "diagnosis label" in row_text:
            header_row = i
            break

    if header_row is None:
        raise ValueError("Cannot find header row containing 'class' and 'Diagnosis label'.")

    df = pd.read_excel(path, sheet_name=0, header=header_row)
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def find_class_col(df):
    for col in df.columns:
        if str(col).strip().lower() == "class":
            return col
    raise ValueError(f"Cannot find class column. Columns: {list(df.columns)}")


def find_readers(df):
    """寻找所�?*_Diagnosis label，并匹配其右侧最近的 Confidence score"""
    cols = list(df.columns)
    readers = []

    for i, col in enumerate(cols):
        col_str = str(col).strip()

        if col_str.endswith("_Diagnosis label"):
            reader_name = col_str.replace("_Diagnosis label", "")
            pred_col = col
            conf_col = None

            for j in range(i + 1, len(cols)):
                next_col = str(cols[j]).strip()

                if next_col.endswith("_Diagnosis label"):
                    break

                if "Confidence" in next_col:
                    conf_col = cols[j]
                    break

            if conf_col is None:
                raise ValueError(f"Cannot find confidence column for reader {reader_name}")

            readers.append((reader_name, pred_col, conf_col))

    if not readers:
        raise ValueError(f"Cannot find reader columns. Columns: {cols}")

    return readers


def compute_metrics(y_true, y_pred, conf):
    y_true = np.array([normalize_label(x) for x in y_true])
    y_pred = np.array([normalize_label(x) for x in y_pred])

    # 兼容 list / pandas Series / numpy array
    conf = np.asarray(pd.to_numeric(pd.Series(conf), errors="coerce"), dtype=float)

    valid = pd.notna(y_true) & pd.notna(y_pred) & pd.notna(conf)
    y_true, y_pred, conf = y_true[valid], y_pred[valid], conf[valid]

    correct = y_true == y_pred

    acc = correct.mean() * 100

    recalls = []
    for c in CLASSES:
        mask = y_true == c
        if np.sum(mask) == 0:
            recall = np.nan
        else:
            recall = np.sum((y_true == c) & (y_pred == c)) / np.sum(mask)
        recalls.append(recall)

    bacc = np.nanmean(recalls) * 100

    cwa = conf[correct].sum() / conf.sum() * 100

    return {
        "ACC": acc,
        "BACC": bacc,
        "CWA": cwa,
        "Correct": int(correct.sum()),
        "Total": int(len(y_true)),
        "Recall": dict(zip(CLASSES, [r * 100 for r in recalls])),
        "y_true": y_true,
        "y_pred": y_pred,
        "conf": conf,
    }

def cohen_kappa(pred1, pred2):
    pred1 = np.array([normalize_label(x) for x in pred1])
    pred2 = np.array([normalize_label(x) for x in pred2])

    valid = pd.notna(pred1) & pd.notna(pred2)
    pred1, pred2 = pred1[valid], pred2[valid]

    po = np.mean(pred1 == pred2)

    pe = 0
    for c in CLASSES:
        pe += np.mean(pred1 == c) * np.mean(pred2 == c)

    kappa = (po - pe) / (1 - pe)
    return kappa, po, pe


def main():
    df = read_excel_with_auto_header(EXCEL_PATH)

    print("=" * 60)
    print(f"Excel path: {EXCEL_PATH}")
    print(f"Columns: {list(df.columns)}")
    print("=" * 60)

    class_col = find_class_col(df)
    readers = find_readers(df)

    print("Detected readers:")
    for name, pred_col, conf_col in readers:
        print(f"{name}: {pred_col} | {conf_col}")

    all_true, all_pred, all_conf = [], [], []
    reader_preds = []

    print("\n" + "=" * 60)
    print("PER-READER RESULTS")
    print("=" * 60)

    for name, pred_col, conf_col in readers:
        metrics = compute_metrics(df[class_col], df[pred_col], df[conf_col])

        print(f"\nReader: {name}")
        print(f"Correct / Total: {metrics['Correct']} / {metrics['Total']}")
        print(f"ACC : {metrics['ACC']:.2f}%")
        print(f"BACC: {metrics['BACC']:.2f}%")
        print(f"CWA : {metrics['CWA']:.2f}%")

        print("Recall:")
        for c in CLASSES:
            print(f"  {c}: {metrics['Recall'][c]:.2f}%")

        all_true.extend(metrics["y_true"])
        all_pred.extend(metrics["y_pred"])
        all_conf.extend(metrics["conf"])
        reader_preds.append(metrics["y_pred"])

    pooled = compute_metrics(all_true, all_pred, all_conf)

    print("\n" + "=" * 60)
    print("POOLED RESULTS")
    print("=" * 60)
    print(f"Correct / Total: {pooled['Correct']} / {pooled['Total']}")
    print(f"ACC : {pooled['ACC']:.2f}%")
    print(f"BACC: {pooled['BACC']:.2f}%")
    print(f"CWA : {pooled['CWA']:.2f}%")

    print("Recall:")
    for c in CLASSES:
        print(f"  {c}: {pooled['Recall'][c]:.2f}%")

    if len(reader_preds) == 2:
        kappa, po, pe = cohen_kappa(reader_preds[0], reader_preds[1])
        print("\n" + "=" * 60)
        print("COHEN'S KAPPA")
        print("=" * 60)
        print(f"Observed agreement p_o : {po:.4f}")
        print(f"Expected agreement p_e : {pe:.4f}")
        print(f"Kappa                  : {kappa:.4f}")
    else:
        kappa = None
        print("\nKappa cannot be calculated because reader number is not 2.")

    method_name = os.path.splitext(os.path.basename(EXCEL_PATH))[0]

    print("\n" + "=" * 60)
    print("LATEX TABLE ROW")
    print("=" * 60)

    if kappa is not None:
        print(
            f"MRI+Syn PET({method_name}) & "
            f"{pooled['ACC']:.2f} & "
            f"{pooled['BACC']:.2f} & "
            f"{pooled['CWA']:.2f} & "
            f"{kappa:.2f} \\\\"
        )
    else:
        print(
            f"MRI+Syn PET({method_name}) & "
            f"{pooled['ACC']:.2f} & "
            f"{pooled['BACC']:.2f} & "
            f"{pooled['CWA']:.2f} & "
            f"-- \\\\"
        )


if __name__ == "__main__":
    main()