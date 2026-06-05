import os
import numpy as np
import nibabel as nib


def convert_overlap_volumes_to_nii(input_dir, output_dir, reference_root):
    os.makedirs(output_dir, exist_ok=True)

    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".npy"):
            continue

        volume_id = os.path.splitext(fname)[0]
        volume_path = os.path.join(input_dir, fname)

        volume = np.load(volume_path).astype(np.float32)

        ref_path = os.path.join(reference_root, volume_id + ".nii.gz")
        if os.path.exists(ref_path):
            ref_img = nib.load(ref_path)
            affine = ref_img.affine
            header = ref_img.header.copy()
        else:
            affine = np.eye(4)
            header = None
            print(f"WARNING: reference not found for {volume_id}, using identity affine.")

        nii_img = nib.Nifti1Image(volume, affine, header=header)
        save_path = os.path.join(output_dir, volume_id + ".nii.gz")
        nib.save(nii_img, save_path)

        print(f"Saved {save_path}, shape={volume.shape}")


input_dir = "/path/to/2.5D_128/logs/experiment/version_4/test/test_samples/pred_volumes_overlap_avg"
output_dir = "/path/to/2.5D_128/logs/experiment/version_4/test/test_samples/pred_overlap_avg_nii"
reference_root = "/path/to/classify/datasets/PET_99_test"

convert_overlap_volumes_to_nii(input_dir, output_dir, reference_root)
