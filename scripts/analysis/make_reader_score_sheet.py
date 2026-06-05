import os
import pandas as pd


def get_case_id(fname):
    """
    支持�?
    case1.gz
    case1.nii.gz
    case1.nii
    """
    name = fname
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    elif name.endswith(".gz"):
        name = name[:-3]
    return name


def make_reader_score_sheet(
    session_dir,
    save_csv,
    reader_ids=("Reader1", "Reader2", "Reader3"),
    session_name="Session_1",
    condition_code="A"
):
    mri_dir = os.path.join(session_dir, "MRI")
    pet_dir = os.path.join(session_dir, "PET")

    if not os.path.isdir(mri_dir):
        raise RuntimeError(f"MRI folder not found: {mri_dir}")
    if not os.path.isdir(pet_dir):
        raise RuntimeError(f"PET folder not found: {pet_dir}")

    mri_files = {}
    for fname in os.listdir(mri_dir):
        if fname.startswith("."):
            continue
        case_id = get_case_id(fname)
        mri_files[case_id] = os.path.join(mri_dir, fname)

    pet_files = {}
    for fname in os.listdir(pet_dir):
        if fname.startswith("."):
            continue
        case_id = get_case_id(fname)
        pet_files[case_id] = os.path.join(pet_dir, fname)

    common_cases = sorted(set(mri_files.keys()) & set(pet_files.keys()))

    if len(common_cases) == 0:
        raise RuntimeError("No paired MRI/PET cases found.")

    missing_mri = sorted(set(pet_files.keys()) - set(mri_files.keys()))
    missing_pet = sorted(set(mri_files.keys()) - set(pet_files.keys()))

    print(f"Paired cases found: {len(common_cases)}")
    if missing_mri:
        print(f"Cases missing MRI: {missing_mri}")
    if missing_pet:
        print(f"Cases missing PET: {missing_pet}")

    records = []

    for reader_id in reader_ids:
        for case_id in common_cases:
            records.append({
                "Reader_ID": reader_id,
                "Session": session_name,
                "Case_ID": case_id,
                "MRI_File": mri_files[case_id],
                "PET_File": pet_files[case_id],

                # 给你后续统计用，不建议医生知道真实含�?
                "Condition": condition_code,

                # 下面由医生填�?
                "Diagnosis": "",              # CN / MCI / AD
                "Confidence": "",             # 1 / 2 / 3 / 4 / 5
                "Clinically_Acceptable": "",  # Yes / No
                "Comments": ""
            })

    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(save_csv), exist_ok=True)
    df.to_csv(save_csv, index=False, encoding="utf-8-sig")

    print(f"Saved score sheet to: {save_csv}")
    print(f"Total rows: {len(df)}")
    print(f"Cases: {len(common_cases)}")
    print(f"Readers: {len(reader_ids)}")


if __name__ == "__main__":
    session_dir = "/path/to/Clinical_reader_study/Session_1"
    save_csv = "/path/to/Clinical_reader_study/reader_score_sheet_session1.csv"

    make_reader_score_sheet(
        session_dir=session_dir,
        save_csv=save_csv,
        reader_ids=("Reader1", "Reader2", "Reader3"),
        session_name="Session_1",
        condition_code="A"
    )