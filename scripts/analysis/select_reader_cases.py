import os
import csv
import random
import shutil


# ============================================================
# 1. 源数据路�?
# ============================================================
SRC_MRI_DIR = "/path/to/clicinal/MRI"
SRC_GT_PET_DIR = "/path/to/clicinal/SelfRDB_PET"
SRC_OURS_PET_DIR = "/path/to/clicinal/Ours_PET"


# ============================================================
# 2. 输出路径
# ============================================================
READER_ROOT = "/path/to/clicinal/reader_4"

# reader_1/2：MRI + GT_PET
GROUP2_ROOT = os.path.join(READER_ROOT, "4")
GROUP2_MRI_DIR = os.path.join(GROUP2_ROOT, "MRI")
GROUP2_MRI_PET_DIR = os.path.join(GROUP2_ROOT, "MRI_PET")

# reader_1/3：MRI + Ours_PET
GROUP3_ROOT = os.path.join(READER_ROOT, "3")
GROUP3_MRI_DIR = os.path.join(GROUP3_ROOT, "MRI")
GROUP3_MRI_PET_DIR = os.path.join(GROUP3_ROOT, "MRI_PET")


# ============================================================
# 3. 划分结果保存路径
# ============================================================
CSV_SAVE_PATH = os.path.join(READER_ROOT, "reader_4_case_split.csv")
TXT_SAVE_PATH = os.path.join(READER_ROOT, "reader_4_case_split.txt")


# ============================================================
# 4. 随机种子
#    固定后，每次运行得到的划分和case顺序都一�?
#    想重新随机划分，只需改成另一个数�?
# ============================================================
RANDOM_SEED = 42


# ============================================================
# 5. 是否清空旧输出目�?
#    True：删除旧�?reader_1/2 �?reader_1/3 目录后重新构�?
# ============================================================
CLEAN_OLD_OUTPUT = True


# ============================================================
# 6. 各类别候�?ID
# ============================================================
AD_IDS = [
    "020_S_0213_6",
    "019_S_4477_3",
    "037_S_0627_4",
    "024_S_1171_5",
    "053_S_1044_6",
    "011_S_0003_4",
    "005_S_0221_2",
    "020_S_0213",
    "011_S_0183_2",
]

CN_IDS = [
    "036_S_0672_3",
    "036_S_0672",
    "014_S_4577",
    "128_S_0272_4",
    "941_S_4100",
    "010_S_0419_2",
    "036_S_0672_5",
    "010_S_0419",
    "127_S_4198_2",
]

MCI_IDS = [
    "006_S_1130_4",
    "006_S_1130_13",
    "005_S_0222_12",
    "070_S_6911_2",
    "130_S_4294_3",
    "007_S_0101_4",
]


# ============================================================
# 7. 两组病例数量设置
# ============================================================
# reader_1/2：GT_PET �?
GROUP2_COUNTS = {
    "AD": 4,
    "CN": 5,
    "MCI": 3,
}

# reader_1/3：Ours_PET �?
GROUP3_COUNTS = {
    "AD": 5,
    "CN": 4,
    "MCI": 3,
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def reset_dir(path):
    """
    删除旧目录后重新创建�?
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def split_one_class(id_list, group2_num):
    """
    对某一类别进行不重叠划分：
    - group2_num 个进�?reader_1/2
    - 剩余全部进入 reader_1/3
    """
    id_list = id_list.copy()
    random.shuffle(id_list)

    group2_ids = id_list[:group2_num]
    group3_ids = id_list[group2_num:]

    return group2_ids, group3_ids


def build_item_list(ids, cls_name, pet_group):
    """
    构造结构化病例记录�?
    pet_group:
        - "GT_PET"
        - "Ours_PET"
    """
    items = []
    for case_id in ids:
        items.append({
            "Class": cls_name,
            "ID": case_id,
            "PET_Group": pet_group,
        })
    return items


def check_source_files(items, pet_src_dir, group_name):
    """
    检�?MRI �?PET 源文件是否都存在�?
    """
    missing = []

    for item in items:
        case_id = item["ID"]

        mri_path = os.path.join(SRC_MRI_DIR, f"{case_id}.nii.gz")
        pet_path = os.path.join(pet_src_dir, f"{case_id}.nii.gz")

        if not os.path.exists(mri_path) or not os.path.exists(pet_path):
            missing.append({
                "Group": group_name,
                "ID": case_id,
                "MRI_missing": not os.path.exists(mri_path),
                "PET_missing": not os.path.exists(pet_path),
                "MRI_path": mri_path,
                "PET_path": pet_path,
            })

    return missing


def copy_group_data(
    items,
    pet_src_dir,
    out_mri_dir,
    out_mri_pet_dir,
    group_label
):
    """
    复制一组数据，并按case编号重新命名�?
    输出结构�?
    - MRI/caseX.nii.gz
    - MRI_PET/caseX/MRI.nii.gz
    - MRI_PET/caseX/PET.nii.gz
    """
    copied_records = []

    for idx, item in enumerate(items, start=1):
        case_name = f"case{idx}"
        original_id = item["ID"]
        cls_name = item["Class"]

        # 源文�?
        src_mri_path = os.path.join(SRC_MRI_DIR, f"{original_id}.nii.gz")
        src_pet_path = os.path.join(pet_src_dir, f"{original_id}.nii.gz")

        # ------------------------------------------------------------
        # 1. 单独 MRI 文件夹中保存：caseX.nii.gz
        # ------------------------------------------------------------
        dst_single_mri_path = os.path.join(out_mri_dir, f"{case_name}.nii.gz")
        shutil.copy2(src_mri_path, dst_single_mri_path)

        # ------------------------------------------------------------
        # 2. MRI_PET/caseX/ 下面保存 MRI.nii.gz �?PET.nii.gz
        # ------------------------------------------------------------
        case_folder = os.path.join(out_mri_pet_dir, case_name)
        ensure_dir(case_folder)

        dst_pair_mri_path = os.path.join(case_folder, "MRI.nii.gz")
        dst_pair_pet_path = os.path.join(case_folder, "PET.nii.gz")

        shutil.copy2(src_mri_path, dst_pair_mri_path)
        shutil.copy2(src_pet_path, dst_pair_pet_path)

        copied_records.append({
            "Reader_Group": group_label,
            "Case_Name": case_name,
            "Class": cls_name,
            "Original_ID": original_id,
            "PET_Group": item["PET_Group"],
            "Single_MRI_Path": dst_single_mri_path,
            "Pair_Folder": case_folder,
        })

        print(
            f"[完成] {group_label} | {case_name} | "
            f"{cls_name} | {original_id}"
        )

    return copied_records


def save_csv(records):
    """
    保存 case 编号、原始ID、类别、所属组等信息�?
    """
    with open(CSV_SAVE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Reader_Group",
            "Case_Name",
            "Class",
            "Original_ID",
            "PET_Group",
            "Single_MRI_Path",
            "Pair_Folder",
        ])

        for rec in records:
            writer.writerow([
                rec["Reader_Group"],
                rec["Case_Name"],
                rec["Class"],
                rec["Original_ID"],
                rec["PET_Group"],
                rec["Single_MRI_Path"],
                rec["Pair_Folder"],
            ])


def save_txt(group2_records, group3_records):
    """
    保存便于人工查看的划分文本�?
    """
    with open(TXT_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("Reader 1 MRI/PET case split\n")
        f.write("=" * 90 + "\n\n")

        # --------------------------------------------------------
        # reader_1/2
        # --------------------------------------------------------
        f.write("reader_1/2 : MRI + GT_PET\n")
        f.write("Target ratio: AD=4, CN=5, MCI=3\n")
        f.write("-" * 90 + "\n")

        for rec in group2_records:
            f.write(
                f"{rec['Case_Name']} = {rec['Original_ID']} "
                f"({rec['Class']})\n"
            )

        f.write("\nCategory summary:\n")
        for cls in ["AD", "CN", "MCI"]:
            cls_items = [
                f"{rec['Case_Name']}:{rec['Original_ID']}"
                for rec in group2_records
                if rec["Class"] == cls
            ]
            f.write(f"{cls} ({len(cls_items)}): " + "�?.join(cls_items) + "\n")

        f.write("\n\n")

        # --------------------------------------------------------
        # reader_1/3
        # --------------------------------------------------------
        f.write("reader_1/3 : MRI + Ours_PET\n")
        f.write("Target ratio: AD=5, CN=4, MCI=3\n")
        f.write("-" * 90 + "\n")

        for rec in group3_records:
            f.write(
                f"{rec['Case_Name']} = {rec['Original_ID']} "
                f"({rec['Class']})\n"
            )

        f.write("\nCategory summary:\n")
        for cls in ["AD", "CN", "MCI"]:
            cls_items = [
                f"{rec['Case_Name']}:{rec['Original_ID']}"
                for rec in group3_records
                if rec["Class"] == cls
            ]
            f.write(f"{cls} ({len(cls_items)}): " + "�?.join(cls_items) + "\n")


def print_group_summary(title, records):
    """
    打印每组的case-ID对应关系�?
    """
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)

    for rec in records:
        print(
            f"{rec['Case_Name']:>6} = {rec['Original_ID']:<18} "
            f"| {rec['Class']}"
        )

    print("-" * 100)
    for cls in ["AD", "CN", "MCI"]:
        cls_records = [r for r in records if r["Class"] == cls]
        print(f"{cls}: {len(cls_records)}")


def main():
    random.seed(RANDOM_SEED)

    # ============================================================
    # 1. 清空并重建输出目�?
    # ============================================================
    ensure_dir(READER_ROOT)

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
    # 2. 按类别不重叠划分
    # ============================================================
    group2_ad, group3_ad = split_one_class(AD_IDS, GROUP2_COUNTS["AD"])
    group2_cn, group3_cn = split_one_class(CN_IDS, GROUP2_COUNTS["CN"])
    group2_mci, group3_mci = split_one_class(MCI_IDS, GROUP2_COUNTS["MCI"])

    # ============================================================
    # 3. 组织病例记录
    # ============================================================
    group2_items = []
    group2_items += build_item_list(group2_ad, "AD", "GT_PET")
    group2_items += build_item_list(group2_cn, "CN", "GT_PET")
    group2_items += build_item_list(group2_mci, "MCI", "GT_PET")

    group3_items = []
    group3_items += build_item_list(group3_ad, "AD", "Ours_PET")
    group3_items += build_item_list(group3_cn, "CN", "Ours_PET")
    group3_items += build_item_list(group3_mci, "MCI", "Ours_PET")

    # ============================================================
    # 4. 对组内病例顺序再次随机打乱，决定 case1-case12 顺序
    #    这样不会出现前几个全是AD、后几个全是CN的情�?
    # ============================================================
    random.shuffle(group2_items)
    random.shuffle(group3_items)

    # ============================================================
    # 5. 检查两�?ID 无重�?
    # ============================================================
    group2_ids = {item["ID"] for item in group2_items}
    group3_ids = {item["ID"] for item in group3_items}

    overlap_ids = group2_ids & group3_ids
    if overlap_ids:
        raise RuntimeError(f"错误：reader_1/2 �?reader_1/3 存在重复ID：{overlap_ids}")

    print("[检查通过] reader_1/2 �?reader_1/3 的病�?ID 完全不重叠�?)

    # ============================================================
    # 6. 检查源 MRI/PET 文件是否存在
    # ============================================================
    missing_group2 = check_source_files(
        group2_items,
        SRC_GT_PET_DIR,
        group_name="reader_1/2"
    )

    missing_group3 = check_source_files(
        group3_items,
        SRC_OURS_PET_DIR,
        group_name="reader_1/3"
    )

    all_missing = missing_group2 + missing_group3

    if all_missing:
        print("\n发现缺失文件，脚本终止，未执行复制：")
        print("=" * 100)
        for item in all_missing:
            print(
                f"[{item['Group']}] ID={item['ID']} | "
                f"MRI缺失={item['MRI_missing']} | "
                f"PET缺失={item['PET_missing']}"
            )
            if item["MRI_missing"]:
                print(f"  MRI: {item['MRI_path']}")
            if item["PET_missing"]:
                print(f"  PET: {item['PET_path']}")

        raise FileNotFoundError("存在缺失�?MRI �?PET 文件，请先检查源数据�?)

    print("[检查通过] 所有选中病例�?MRI �?PET 文件均存在�?)

    # ============================================================
    # 7. 复制数据并构建目�?
    # ============================================================
    print("\n" + "=" * 100)
    print("开始构�?reader_1/2：MRI + GT_PET")
    print("=" * 100)

    group2_records = copy_group_data(
        items=group2_items,
        pet_src_dir=SRC_GT_PET_DIR,
        out_mri_dir=GROUP2_MRI_DIR,
        out_mri_pet_dir=GROUP2_MRI_PET_DIR,
        group_label="reader_1/2"
    )

    print("\n" + "=" * 100)
    print("开始构�?reader_1/3：MRI + Ours_PET")
    print("=" * 100)

    group3_records = copy_group_data(
        items=group3_items,
        pet_src_dir=SRC_OURS_PET_DIR,
        out_mri_dir=GROUP3_MRI_DIR,
        out_mri_pet_dir=GROUP3_MRI_PET_DIR,
        group_label="reader_1/3"
    )

    # ============================================================
    # 8. 保存划分记录
    # ============================================================
    all_records = group2_records + group3_records
    save_csv(all_records)
    save_txt(group2_records, group3_records)

    # ============================================================
    # 9. 输出划分摘要
    # ============================================================
    print_group_summary(
        "reader_1/2：MRI + GT_PET，case编号对应关系",
        group2_records
    )

    print_group_summary(
        "reader_1/3：MRI + Ours_PET，case编号对应关系",
        group3_records
    )

    print("\n" + "=" * 100)
    print("全部完成")
    print("=" * 100)
    print(f"reader_1/2 输出路径: {GROUP2_ROOT}")
    print(f"reader_1/3 输出路径: {GROUP3_ROOT}")
    print(f"病例划分 CSV: {CSV_SAVE_PATH}")
    print(f"病例划分 TXT: {TXT_SAVE_PATH}")
    print(f"reader_1/2 病例�? {len(group2_records)}")
    print(f"reader_1/3 病例�? {len(group3_records)}")


if __name__ == "__main__":
    main()