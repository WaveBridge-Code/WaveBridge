import os
import csv
import shutil


# ============================================================
# 1. 源数据路�?
# ============================================================
SRC_MRI_DIR = "/path/to/clicinal/MRI"
SRC_GT_PET_DIR = "/path/to/clicinal/SelfRDB_PET"
SRC_OURS_PET_DIR = "/path/to/clicinal/Ours_PET"


# ============================================================
# 2. 第一轮划分记�?
# ============================================================
SESSION1_SPLIT_CSV = "/path/to/clicinal/reader_4/reader_4_case_split.csv"


# ============================================================
# 3. 第二轮输出根路径
# 注意：这里按你写的路�?sesssion_2，三�?s
# ============================================================
SESSION2_READER_ROOT = "/path/to/clicinal/sesssion_2/reader_4"

# 第二�?reader_1/2：MRI + GT_PET
GROUP2_ROOT = os.path.join(SESSION2_READER_ROOT, "4")
GROUP2_MRI_DIR = os.path.join(GROUP2_ROOT, "MRI")
GROUP2_MRI_PET_DIR = os.path.join(GROUP2_ROOT, "MRI_PET")

# 第二�?reader_1/3：MRI + Ours_PET
GROUP3_ROOT = os.path.join(SESSION2_READER_ROOT, "3")
GROUP3_MRI_DIR = os.path.join(GROUP3_ROOT, "MRI")
GROUP3_MRI_PET_DIR = os.path.join(GROUP3_ROOT, "MRI_PET")


# ============================================================
# 4. 第二轮划分记录保存路�?
# ============================================================
SESSION2_SPLIT_CSV = os.path.join(SESSION2_READER_ROOT, "reader_4_session2_case_split.csv")
SESSION2_SPLIT_TXT = os.path.join(SESSION2_READER_ROOT, "reader_4_session2_case_split.txt")


# ============================================================
# 5. 是否清空旧的第二轮输�?
# ============================================================
CLEAN_OLD_OUTPUT = True


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def read_session1_split(csv_path):
    """
    读取第一�?reader_1_case_split.csv�?

    需要包含这些列�?
    Reader_Group, Case_Name, Class, Original_ID, PET_Group
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到第一轮划分文�? {csv_path}")

    group2_records = []
    group3_records = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_cols = {"Reader_Group", "Case_Name", "Class", "Original_ID", "PET_Group"}
        missing_cols = required_cols - set(reader.fieldnames)

        if missing_cols:
            raise ValueError(f"CSV 缺少必要�? {missing_cols}")

        for row in reader:
            reader_group = row["Reader_Group"].strip()

            item = {
                "Old_Reader_Group": reader_group,
                "Old_Case_Name": row["Case_Name"].strip(),
                "Class": row["Class"].strip(),
                "Original_ID": row["Original_ID"].strip(),
                "Old_PET_Group": row["PET_Group"].strip(),
            }

            if reader_group == "reader_1/2":
                group2_records.append(item)
            elif reader_group == "reader_1/3":
                group3_records.append(item)
            else:
                print(f"[警告] 未识别的 Reader_Group，跳�? {reader_group}")

    return group2_records, group3_records


def sort_by_case_number(records):
    """
    �?case1, case2, ... 的数字顺序排序�?
    """
    def case_key(x):
        case_name = x["Old_Case_Name"]
        num = int(case_name.replace("case", ""))
        return num

    return sorted(records, key=case_key)


def check_no_overlap(group2_records, group3_records):
    group2_ids = {x["Original_ID"] for x in group2_records}
    group3_ids = {x["Original_ID"] for x in group3_records}

    overlap = group2_ids & group3_ids
    if overlap:
        raise RuntimeError(f"第一�?reader_1/2 �?reader_1/3 存在重复 ID: {overlap}")

    print("[检查通过] 第一�?reader_1/2 �?reader_1/3 �?ID 无重叠�?)


def check_source_files(records, pet_src_dir, target_group):
    """
    检查第二轮要复制的 MRI �?PET 是否存在�?
    """
    missing = []

    for item in records:
        case_id = item["Original_ID"]

        mri_path = os.path.join(SRC_MRI_DIR, f"{case_id}.nii.gz")
        pet_path = os.path.join(pet_src_dir, f"{case_id}.nii.gz")

        if not os.path.exists(mri_path) or not os.path.exists(pet_path):
            missing.append({
                "Target_Group": target_group,
                "ID": case_id,
                "MRI_missing": not os.path.exists(mri_path),
                "PET_missing": not os.path.exists(pet_path),
                "MRI_path": mri_path,
                "PET_path": pet_path,
            })

    return missing


def copy_group_data(records, pet_src_dir, out_mri_dir, out_mri_pet_dir, target_group, pet_group_name):
    """
    构建第二轮数据结构：
    - MRI/caseX.nii.gz
    - MRI_PET/caseX/MRI.nii.gz
    - MRI_PET/caseX/PET.nii.gz

    这里 caseX 顺序沿用原组内部�?case 顺序�?
    """
    copied_records = []

    for idx, item in enumerate(records, start=1):
        new_case_name = f"case{idx}"
        original_id = item["Original_ID"]
        cls_name = item["Class"]

        src_mri_path = os.path.join(SRC_MRI_DIR, f"{original_id}.nii.gz")
        src_pet_path = os.path.join(pet_src_dir, f"{original_id}.nii.gz")

        # 1. 单独 MRI 文件�?
        dst_single_mri_path = os.path.join(out_mri_dir, f"{new_case_name}.nii.gz")
        shutil.copy2(src_mri_path, dst_single_mri_path)

        # 2. MRI_PET/caseX/
        case_folder = os.path.join(out_mri_pet_dir, new_case_name)
        ensure_dir(case_folder)

        dst_pair_mri_path = os.path.join(case_folder, "MRI.nii.gz")
        dst_pair_pet_path = os.path.join(case_folder, "PET.nii.gz")

        shutil.copy2(src_mri_path, dst_pair_mri_path)
        shutil.copy2(src_pet_path, dst_pair_pet_path)

        copied_records.append({
            "New_Reader_Group": target_group,
            "New_Case_Name": new_case_name,
            "Class": cls_name,
            "Original_ID": original_id,
            "New_PET_Group": pet_group_name,
            "Old_Reader_Group": item["Old_Reader_Group"],
            "Old_Case_Name": item["Old_Case_Name"],
            "Old_PET_Group": item["Old_PET_Group"],
            "Single_MRI_Path": dst_single_mri_path,
            "Pair_Folder": case_folder,
        })

        print(
            f"[完成] {target_group} | {new_case_name} | "
            f"{cls_name} | {original_id} | "
            f"from {item['Old_Reader_Group']}/{item['Old_Case_Name']}"
        )

    return copied_records


def save_session2_csv(records):
    with open(SESSION2_SPLIT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "New_Reader_Group",
            "New_Case_Name",
            "Class",
            "Original_ID",
            "New_PET_Group",
            "Old_Reader_Group",
            "Old_Case_Name",
            "Old_PET_Group",
            "Single_MRI_Path",
            "Pair_Folder",
        ])

        for rec in records:
            writer.writerow([
                rec["New_Reader_Group"],
                rec["New_Case_Name"],
                rec["Class"],
                rec["Original_ID"],
                rec["New_PET_Group"],
                rec["Old_Reader_Group"],
                rec["Old_Case_Name"],
                rec["Old_PET_Group"],
                rec["Single_MRI_Path"],
                rec["Pair_Folder"],
            ])


def save_session2_txt(group2_records, group3_records):
    with open(SESSION2_SPLIT_TXT, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("Session 2 / Reader 1 swapped case split\n")
        f.write("=" * 100 + "\n\n")

        f.write("说明：\n")
        f.write("第一�?reader_1/3 �?ID -> 第二�?reader_1/2，使�?GT_PET\n")
        f.write("第一�?reader_1/2 �?ID -> 第二�?reader_1/3，使�?Ours_PET\n\n")

        f.write("=" * 100 + "\n")
        f.write("Session 2 reader_1/2 : MRI + GT_PET\n")
        f.write("=" * 100 + "\n")
        for rec in group2_records:
            f.write(
                f"{rec['New_Case_Name']} = {rec['Original_ID']} "
                f"({rec['Class']}) | old: {rec['Old_Reader_Group']}/{rec['Old_Case_Name']}\n"
            )

        f.write("\nCategory summary:\n")
        for cls in ["AD", "CN", "MCI"]:
            cls_items = [r for r in group2_records if r["Class"] == cls]
            f.write(f"{cls}: {len(cls_items)}\n")

        f.write("\n\n")

        f.write("=" * 100 + "\n")
        f.write("Session 2 reader_1/3 : MRI + Ours_PET\n")
        f.write("=" * 100 + "\n")
        for rec in group3_records:
            f.write(
                f"{rec['New_Case_Name']} = {rec['Original_ID']} "
                f"({rec['Class']}) | old: {rec['Old_Reader_Group']}/{rec['Old_Case_Name']}\n"
            )

        f.write("\nCategory summary:\n")
        for cls in ["AD", "CN", "MCI"]:
            cls_items = [r for r in group3_records if r["Class"] == cls]
            f.write(f"{cls}: {len(cls_items)}\n")


def print_summary(title, records):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for rec in records:
        print(
            f"{rec['New_Case_Name']:>6} = {rec['Original_ID']:<18} "
            f"| {rec['Class']:<3} | old: {rec['Old_Reader_Group']}/{rec['Old_Case_Name']}"
        )

    print("-" * 100)
    for cls in ["AD", "CN", "MCI"]:
        cls_records = [r for r in records if r["Class"] == cls]
        print(f"{cls}: {len(cls_records)}")


def main():
    # ============================================================
    # 1. 读取第一轮划�?
    # ============================================================
    old_group2_records, old_group3_records = read_session1_split(SESSION1_SPLIT_CSV)

    old_group2_records = sort_by_case_number(old_group2_records)
    old_group3_records = sort_by_case_number(old_group3_records)

    print(f"第一�?reader_1/2 病例�? {len(old_group2_records)}")
    print(f"第一�?reader_1/3 病例�? {len(old_group3_records)}")

    check_no_overlap(old_group2_records, old_group3_records)

    # ============================================================
    # 2. 第二轮互换逻辑
    # ============================================================
    # 第一�?reader_1/3 -> 第二�?reader_1/2，使�?GT_PET
    session2_group2_items = old_group3_records

    # 第一�?reader_1/2 -> 第二�?reader_1/3，使�?Ours_PET
    session2_group3_items = old_group2_records

    # ============================================================
    # 3. 清空并重建输出目�?
    # ============================================================
    ensure_dir(SESSION2_READER_ROOT)

    if CLEAN_OLD_OUTPUT:
        reset_dir(GROUP2_ROOT)
        reset_dir(GROUP3_ROOT)
    else:
        ensure_dir(GROUP2_ROOT)
        ensure_dir(GROUP3_ROOT)

    ensure_dir(GROUP2_MRI_DIR)
    ensure_dir(GROUP2_MRI_PET_DIR)
    ensure_dir(GROUP3_MRI_DIR)
    ensure_dir(GROUP3_MRI_PET_DIR)

    # ============================================================
    # 4. 检查源文件是否存在
    # ============================================================
    missing_group2 = check_source_files(
        session2_group2_items,
        SRC_GT_PET_DIR,
        target_group="session2/reader_1/2"
    )

    missing_group3 = check_source_files(
        session2_group3_items,
        SRC_OURS_PET_DIR,
        target_group="session2/reader_1/3"
    )

    all_missing = missing_group2 + missing_group3

    if all_missing:
        print("\n发现缺失文件，脚本终止，未执行复制：")
        print("=" * 100)
        for item in all_missing:
            print(
                f"[{item['Target_Group']}] ID={item['ID']} | "
                f"MRI缺失={item['MRI_missing']} | "
                f"PET缺失={item['PET_missing']}"
            )
            if item["MRI_missing"]:
                print(f"  MRI: {item['MRI_path']}")
            if item["PET_missing"]:
                print(f"  PET: {item['PET_path']}")

        raise FileNotFoundError("存在缺失�?MRI �?PET 文件，请先检查源数据�?)

    print("[检查通过] 第二轮所有需要复制的 MRI �?PET 文件均存在�?)

    # ============================================================
    # 5. 复制第二�?reader_1/2：MRI + GT_PET
    # ============================================================
    print("\n" + "=" * 100)
    print("开始构�?Session 2 reader_1/2：MRI + GT_PET")
    print("=" * 100)

    group2_copied_records = copy_group_data(
        records=session2_group2_items,
        pet_src_dir=SRC_GT_PET_DIR,
        out_mri_dir=GROUP2_MRI_DIR,
        out_mri_pet_dir=GROUP2_MRI_PET_DIR,
        target_group="reader_1/2",
        pet_group_name="GT_PET"
    )

    # ============================================================
    # 6. 复制第二�?reader_1/3：MRI + Ours_PET
    # ============================================================
    print("\n" + "=" * 100)
    print("开始构�?Session 2 reader_1/3：MRI + Ours_PET")
    print("=" * 100)

    group3_copied_records = copy_group_data(
        records=session2_group3_items,
        pet_src_dir=SRC_OURS_PET_DIR,
        out_mri_dir=GROUP3_MRI_DIR,
        out_mri_pet_dir=GROUP3_MRI_PET_DIR,
        target_group="reader_1/3",
        pet_group_name="Ours_PET"
    )

    # ============================================================
    # 7. 保存第二轮划分记�?
    # ============================================================
    all_records = group2_copied_records + group3_copied_records

    save_session2_csv(all_records)
    save_session2_txt(group2_copied_records, group3_copied_records)

    # ============================================================
    # 8. 打印摘要
    # ============================================================
    print_summary(
        "Session 2 reader_1/2：MRI + GT_PET，来自第一�?reader_1/3 �?ID",
        group2_copied_records
    )

    print_summary(
        "Session 2 reader_1/3：MRI + Ours_PET，来自第一�?reader_1/2 �?ID",
        group3_copied_records
    )

    print("\n" + "=" * 100)
    print("Session 2 构建完成")
    print("=" * 100)
    print(f"输出根路�? {SESSION2_READER_ROOT}")
    print(f"reader_1/2 输出路径: {GROUP2_ROOT}")
    print(f"reader_1/3 输出路径: {GROUP3_ROOT}")
    print(f"第二�?CSV: {SESSION2_SPLIT_CSV}")
    print(f"第二�?TXT: {SESSION2_SPLIT_TXT}")
    print(f"reader_1/2 病例�? {len(group2_copied_records)}")
    print(f"reader_1/3 病例�? {len(group3_copied_records)}")


if __name__ == "__main__":
    main()