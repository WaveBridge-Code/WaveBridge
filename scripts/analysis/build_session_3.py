import os
import random
import shutil
import pandas as pd


# =========================
# 路径设置
# =========================
# 原始 MRI 路径
mri_dir = "/path/to/clicinal/MRI"

# 原始 Ours 合成 PET 路径
pet_dir = "/path/to/clicinal/Ours"

# 之前保存�?ID-label-Age-Gender 信息�?
info_excel_path = "/path/to/clicinal/MRI_GT_PET_IDs.xlsx"

# �?session 保存路径：Ours
save_root = "/path/to/clicinal/session_3/Ours"

# MRI-only 保存路径
save_mri_dir = os.path.join(save_root, "MRI")

# MRI+Ours PET 保存路径
save_mri_pet_dir = os.path.join(save_root, "MRI+PET")

# case 与原�?ID、label、Age、Gender 的映射表
save_excel_path = os.path.join(save_root, "case_mapping.xlsx")


# =========================
# 固定随机种子
# 注意：这里换成了不同�?GT_PET、D2Diff_PET、SelfRDB_PET 的随机种�?
# =========================
random_seed = 1029
random.seed(random_seed)


# =========================
# 创建保存文件�?
# =========================
os.makedirs(save_mri_dir, exist_ok=True)
os.makedirs(save_mri_pet_dir, exist_ok=True)


# =========================
# 读取 ID-label-Age-Gender 信息
# =========================
info_df = pd.read_excel(info_excel_path)

required_cols = ["ID", "label", "Age", "Gender"]
missing_cols = [col for col in required_cols if col not in info_df.columns]

if missing_cols:
    raise ValueError(f"Excel 中缺少这些列: {missing_cols}")

# 防止 ID 类型不一致导致匹配失�?
info_df["ID"] = info_df["ID"].astype(str)

# 建立 ID -> label / Age / Gender 的映�?
info_dict = info_df.set_index("ID")[["label", "Age", "Gender"]].to_dict(orient="index")

print(f"已读取病例信息数�? {len(info_dict)}")


# =========================
# 读取 .nii.gz 文件
# =========================
def get_nii_files(folder):
    files = []

    for filename in os.listdir(folder):
        if filename.endswith(".nii.gz"):
            files.append(filename)

    return sorted(files)


mri_files = get_nii_files(mri_dir)
pet_files = get_nii_files(pet_dir)


# =========================
# 检�?MRI �?Ours PET 文件数量
# =========================
print(f"MRI 文件数量: {len(mri_files)}")
print(f"Ours PET 文件数量: {len(pet_files)}")

if len(mri_files) != len(pet_files):
    raise ValueError("MRI �?Ours PET 文件数量不一致，请检查数据�?)


# =========================
# 检�?MRI �?Ours PET 是否一一配对
# 要求两个文件夹下�?.nii.gz 文件名完全一�?
# =========================
mri_set = set(mri_files)
pet_set = set(pet_files)

if mri_set != pet_set:
    only_mri = sorted(list(mri_set - pet_set))
    only_pet = sorted(list(pet_set - mri_set))

    print("只存在于 MRI 文件夹中的文�?")
    print(only_mri)

    print("只存在于 Ours PET 文件夹中的文�?")
    print(only_pet)

    raise ValueError("MRI �?Ours PET 文件名不完全一致，请先检查配对关系�?)


# =========================
# 固定随机顺序打乱
# =========================
paired_files = sorted(mri_files)
random.shuffle(paired_files)


# =========================
# 复制并重命名
# =========================
records = []
missing_info_ids = []

for idx, filename in enumerate(paired_files, start=1):
    case_name = f"case{idx}"

    src_mri_path = os.path.join(mri_dir, filename)
    src_pet_path = os.path.join(pet_dir, filename)

    # 原始 ID，去�?.nii.gz
    original_id = filename.replace(".nii.gz", "")

    # 根据 ID 匹配 label / Age / Gender
    if original_id in info_dict:
        label = info_dict[original_id]["label"]
        age = info_dict[original_id]["Age"]
        gender = info_dict[original_id]["Gender"]
    else:
        label = ""
        age = ""
        gender = ""
        missing_info_ids.append(original_id)

    # 1) MRI 文件夹：只保�?MRI，命名为 case1.nii.gz, case2.nii.gz ...
    dst_mri_path = os.path.join(save_mri_dir, f"{case_name}.nii.gz")
    shutil.copy2(src_mri_path, dst_mri_path)

    # 2) MRI+PET 文件夹：每个 case 一个子文件�?
    #    子文件夹内保�?MRI.nii.gz �?PET.nii.gz
    #    这里�?PET �?Ours 合成 PET
    case_dir = os.path.join(save_mri_pet_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)

    dst_case_mri_path = os.path.join(case_dir, "MRI.nii.gz")
    dst_case_pet_path = os.path.join(case_dir, "PET.nii.gz")

    shutil.copy2(src_mri_path, dst_case_mri_path)
    shutil.copy2(src_pet_path, dst_case_pet_path)

    records.append({
        "case": case_name,
        "ID": original_id,
        "label": label,
        "Age": age,
        "Gender": gender,
        "PET_type": "Ours",
        "original_filename": filename,
        "source_MRI_path": src_mri_path,
        "source_Ours_PET_path": src_pet_path,
        "saved_MRI_only_path": dst_mri_path,
        "saved_MRI_PET_MRI_path": dst_case_mri_path,
        "saved_MRI_PET_PET_path": dst_case_pet_path
    })


# =========================
# 保存 Excel 映射�?
# =========================
df = pd.DataFrame(records)
df.to_excel(save_excel_path, index=False)


# =========================
# 输出信息
# =========================
print("=" * 60)
print("处理完成")
print(f"PET 类型: Ours")
print(f"随机种子: {random_seed}")
print(f"总病例数: {len(records)}")
print(f"MRI-only 保存路径: {save_mri_dir}")
print(f"MRI+Ours PET 保存路径: {save_mri_pet_dir}")
print(f"映射 Excel 保存路径: {save_excel_path}")

if missing_info_ids:
    print(f"警告：有 {len(missing_info_ids)} �?ID �?{info_excel_path} 中没有找到对应信�?")
    for mid in missing_info_ids:
        print(mid)
else:
    print("所有病例都成功匹配�?label / Age / Gender")

print("=" * 60)