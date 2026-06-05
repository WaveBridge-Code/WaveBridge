import os
import shutil

# ============================================================
# 路径设置
# ============================================================
ROOT_DIR = "/path/to/Clinical_reader_study"

SRC_MRI_DIR = os.path.join(ROOT_DIR, "MRI")
SRC_PET_DIR = os.path.join(ROOT_DIR, "Ours_PET")

# 新构建的总文件夹
OUT_ROOT = os.path.join(ROOT_DIR, "MRI+Ours_PET")

# 子文件夹 1：单�?MRI，重命名�?case1.nii.gz ...
OUT_MRI_DIR = os.path.join(OUT_ROOT, "MRI")

# 子文件夹 2：MRI + Ours_PET 配对病例文件�?
OUT_PAIR_DIR = os.path.join(OUT_ROOT, "MRI+Ours_PET")


# ============================================================
# 新的随机顺序映射：case编号 -> 原始病例ID
# ============================================================
CASE_MAP = {
    "case1":  "941_S_4100",
    "case2":  "014_S_4577",
    "case3":  "036_S_0672",
    "case4":  "036_S_0672_5",
    "case5":  "036_S_0672_3",
    "case6":  "010_S_0419_2",
    "case7":  "019_S_4477_3",
    "case8":  "020_S_0213_6",
    "case9":  "053_S_1044_6",
    "case10": "005_S_0221_2",
    "case11": "070_S_6911_2",
    "case12": "005_S_0222_12",
    "case13": "037_S_0627_4",
    "case14": "024_S_1171_5",
    "case15": "006_S_1130_4",
    "case16": "006_S_1130_13",
    "case17": "010_S_0419",
    "case18": "128_S_0272_4",
    "case19": "011_S_0003_4",
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    # ============================================================
    # 创建目标目录
    # ============================================================
    ensure_dir(OUT_MRI_DIR)
    ensure_dir(OUT_PAIR_DIR)

    success_count = 0
    missing_files = []

    print("=" * 90)
    print("开始构�?MRI+Ours_PET 条件目录")
    print("=" * 90)

    for case_name, subject_id in CASE_MAP.items():
        # 原始 MRI / Ours PET 文件路径
        src_mri_path = os.path.join(SRC_MRI_DIR, f"{subject_id}.nii.gz")
        src_pet_path = os.path.join(SRC_PET_DIR, f"{subject_id}.nii.gz")

        # 检查源文件是否存在
        mri_exists = os.path.exists(src_mri_path)
        pet_exists = os.path.exists(src_pet_path)

        if not mri_exists or not pet_exists:
            missing_files.append({
                "case": case_name,
                "id": subject_id,
                "MRI_missing": not mri_exists,
                "PET_missing": not pet_exists,
            })

            print(f"[跳过] {case_name} -> {subject_id}")
            if not mri_exists:
                print(f"       缺少 MRI: {src_mri_path}")
            if not pet_exists:
                print(f"       缺少 Ours_PET: {src_pet_path}")
            continue

        # ============================================================
        # 1. 复制 MRI �?OUT_ROOT/MRI，并重命名为 caseX.nii.gz
        # ============================================================
        dst_mri_single_path = os.path.join(OUT_MRI_DIR, f"{case_name}.nii.gz")
        shutil.copy2(src_mri_path, dst_mri_single_path)

        # ============================================================
        # 2. 创建 OUT_ROOT/MRI+Ours_PET/caseX/
        #    并复�?MRI / PET，分别重命名�?MRI.nii.gz / PET.nii.gz
        # ============================================================
        case_folder = os.path.join(OUT_PAIR_DIR, case_name)
        ensure_dir(case_folder)

        dst_case_mri_path = os.path.join(case_folder, "MRI.nii.gz")
        dst_case_pet_path = os.path.join(case_folder, "PET.nii.gz")

        shutil.copy2(src_mri_path, dst_case_mri_path)
        shutil.copy2(src_pet_path, dst_case_pet_path)

        success_count += 1

        print(f"[完成] {case_name} -> {subject_id}")
        print(f"       MRI单独文件: {dst_mri_single_path}")
        print(f"       配对文件�?   {case_folder}")

    # ============================================================
    # 汇总输�?
    # ============================================================
    print("\n" + "=" * 90)
    print("构建完成")
    print("=" * 90)
    print(f"成功构建病例�? {success_count} / {len(CASE_MAP)}")
    print(f"输出总目�? {OUT_ROOT}")

    if missing_files:
        print("\n以下病例存在缺失文件，未完成复制�?)
        for item in missing_files:
            print(
                f"  {item['case']} -> {item['id']} | "
                f"MRI缺失: {item['MRI_missing']} | "
                f"Ours_PET缺失: {item['PET_missing']}"
            )
    else:
        print("\n所�?19 个病例均已成功复制并重命名�?)


if __name__ == "__main__":
    main()