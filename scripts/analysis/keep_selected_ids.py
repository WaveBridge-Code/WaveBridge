import os

# =========================
# 根目�?
# =========================
ROOT_DIR = "/path/to/clicinal"

# 需要处理的五个文件�?
SUB_DIRS = [
    "D2Diff_PET",
    "GT_PET",
    "MRI",
    "Ours_PET",
    "SelfRDB_PET",
]

# =========================
# 需要保留的 ID
# =========================
KEEP_IDS = {
    "020_S_0213_6",
    "019_S_4477_3",
    "037_S_0627_4",
    "024_S_1171_5",
    "053_S_1044_6",
    "011_S_0003_4",
    "005_S_0221_2",
    "036_S_0672_3",
    "036_S_0672",
    "014_S_4577",
    "128_S_0272_4",
    "941_S_4100",
    "010_S_0419_2",
    "036_S_0672_5",
    "010_S_0419",
    "006_S_1130_4",
    "006_S_1130_13",
    "005_S_0222_12",
    "070_S_6911_2",
    "020_S_0213",
    "011_S_0183_2",
    "127_S_4198_2",
    "130_S_4294_3",
    "007_S_0101_4",
}


def get_id_from_filename(filename):
    """
    默认文件名格式为�?
    020_S_0213_6.nii.gz
    """
    if filename.endswith(".nii.gz"):
        return filename[:-7]  # 去掉 .nii.gz
    return None


def main():
    for sub_dir in SUB_DIRS:
        folder_path = os.path.join(ROOT_DIR, sub_dir)

        if not os.path.isdir(folder_path):
            print(f"[跳过] 文件夹不存在: {folder_path}")
            continue

        kept_files = []
        deleted_files = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # 只处理文件，并且只处�?.nii.gz
            if not os.path.isfile(file_path):
                continue
            if not filename.endswith(".nii.gz"):
                continue

            case_id = get_id_from_filename(filename)

            if case_id in KEEP_IDS:
                kept_files.append(filename)
            else:
                os.remove(file_path)
                deleted_files.append(filename)

        print("\n" + "=" * 80)
        print(f"文件�? {sub_dir}")
        print(f"保留文件�? {len(kept_files)}")
        for f in sorted(kept_files):
            print(f"  [保留] {f}")

        print(f"\n删除文件�? {len(deleted_files)}")
        for f in sorted(deleted_files):
            print(f"  [删除] {f}")


if __name__ == "__main__":
    main()