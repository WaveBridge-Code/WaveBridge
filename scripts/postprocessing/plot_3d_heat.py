import glob
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


# ===============================
# Input / output paths
# ===============================
IN_DIR = "/path/to/2.5D_128/logs/experiment/version_4/test/test_samples/pred_overlap_avg_nii/"
OUT_DIR = "/path/to/2.5D_128/logs/experiment/version_4/test/test_samples/heat_3d_nii/"


# ===============================
# Heat-map settings
# ===============================
CMAP = "hot"
DISPLAY_VMIN = 0.0
DISPLAY_VMAX = 1.0

# If your NIfTI values are already in [0, 1], set INPUT_MIN=0.0 and INPUT_MAX=1.0.
# If your prediction values are in [-1, 1], keep the default below.
INPUT_MIN = -1.0
INPUT_MAX = 1.0


# ===============================
# Output controls
# ===============================
# Scalar NIfTI: one 3D volume in [0, 1].
# Open it in 3D Slicer / MRIcroGL / ITK-SNAP and choose the "hot" colormap.
SAVE_SCALAR_HEAT_NII = True

# RGB NIfTI: one 4D volume with RGB channels already colorized by matplotlib hot.
# This is closer to "open and see the heat-map colors", but some medical viewers
# handle RGB/vector NIfTI differently, so keep scalar output too.
SAVE_RGB_HEAT_NII = True

# Optional quick-look middle slice PNGs.
SAVE_MID_SLICE_PREVIEW = True
DPI = 300


def get_subject_id_from_path(nii_path: str) -> str:
    base = os.path.basename(nii_path)
    if base.endswith(".nii.gz"):
        return base[:-7]
    return os.path.splitext(base)[0]


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    vol = np.clip(vol, INPUT_MIN, INPUT_MAX)
    vol = (vol - INPUT_MIN) / (INPUT_MAX - INPUT_MIN)
    return np.clip(vol, 0.0, 1.0)


def apply_colormap_to_volume(vol_01: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap(CMAP)
    rgb = cmap(vol_01, bytes=True)[..., :3]
    return rgb.astype(np.uint8)


def save_scalar_heat_nii(vol_01: np.ndarray, src_nii: nib.Nifti1Image, out_path: str):
    header = src_nii.header.copy()
    header.set_data_dtype(np.float32)

    out_nii = nib.Nifti1Image(
        vol_01.astype(np.float32),
        affine=src_nii.affine,
        header=header,
    )
    nib.save(out_nii, out_path)


def save_rgb_heat_nii(vol_01: np.ndarray, src_nii: nib.Nifti1Image, out_path: str):
    rgb = apply_colormap_to_volume(vol_01)

    header = src_nii.header.copy()
    header.set_data_dtype(np.uint8)

    # Shape: X, Y, Z, 3. Many viewers read this as a vector/RGB volume.
    out_nii = nib.Nifti1Image(
        rgb,
        affine=src_nii.affine,
        header=header,
    )
    nib.save(out_nii, out_path)


def save_png(sl2d: np.ndarray, save_path: str):
    h, w = sl2d.shape
    fig = plt.figure(frameon=False)
    fig.set_size_inches(w / DPI, h / DPI)

    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(
        sl2d,
        cmap=CMAP,
        vmin=DISPLAY_VMIN,
        vmax=DISPLAY_VMAX,
        interpolation="nearest",
    )
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)


def save_mid_slice_preview(vol_01: np.ndarray, out_subj_dir: str, subj_id: str):
    x_mid = vol_01.shape[0] // 2
    y_mid = vol_01.shape[1] // 2
    z_mid = vol_01.shape[2] // 2

    preview_dir = os.path.join(out_subj_dir, "preview_mid_slices")
    os.makedirs(preview_dir, exist_ok=True)

    save_png(np.rot90(vol_01[:, :, z_mid]), os.path.join(preview_dir, f"{subj_id}_axial_z{z_mid:03d}.png"))
    save_png(np.rot90(vol_01[:, y_mid, :]), os.path.join(preview_dir, f"{subj_id}_coronal_y{y_mid:03d}.png"))
    save_png(np.rot90(vol_01[x_mid, :, :]), os.path.join(preview_dir, f"{subj_id}_sagittal_x{x_mid:03d}.png"))


def write_info(
    info_path: str,
    nii_path: str,
    raw_shape,
    raw_min: float,
    raw_max: float,
    scalar_path: str,
    rgb_path: str,
):
    with open(info_path, "w") as f:
        f.write(f"Input file: {nii_path}\n")
        f.write(f"Shape: {raw_shape}\n")
        f.write(f"Raw min: {raw_min:.6f}\n")
        f.write(f"Raw max: {raw_max:.6f}\n")
        f.write(f"Normalize input min: {INPUT_MIN:.6f}\n")
        f.write(f"Normalize input max: {INPUT_MAX:.6f}\n")
        f.write(f"Colormap: {CMAP}\n")
        f.write(f"Scalar heat NIfTI: {scalar_path}\n")
        f.write(f"RGB heat NIfTI: {rgb_path}\n")


def process_one_nii(nii_path: str):
    subj_id = get_subject_id_from_path(nii_path)
    out_subj_dir = os.path.join(OUT_DIR, subj_id)
    os.makedirs(out_subj_dir, exist_ok=True)

    src_nii = nib.load(nii_path)
    vol = np.squeeze(src_nii.get_fdata().astype(np.float32))

    if vol.ndim != 3:
        raise ValueError(f"{nii_path} is not 3D after squeeze, got shape: {vol.shape}")

    raw_min = float(np.nanmin(vol))
    raw_max = float(np.nanmax(vol))
    vol_01 = normalize_volume(vol)

    scalar_path = ""
    rgb_path = ""

    if SAVE_SCALAR_HEAT_NII:
        scalar_path = os.path.join(out_subj_dir, f"{subj_id}_heat_scalar.nii.gz")
        save_scalar_heat_nii(vol_01, src_nii, scalar_path)

    if SAVE_RGB_HEAT_NII:
        rgb_path = os.path.join(out_subj_dir, f"{subj_id}_heat_rgb.nii.gz")
        save_rgb_heat_nii(vol_01, src_nii, rgb_path)

    if SAVE_MID_SLICE_PREVIEW:
        save_mid_slice_preview(vol_01, out_subj_dir, subj_id)

    write_info(
        info_path=os.path.join(out_subj_dir, "info.txt"),
        nii_path=nii_path,
        raw_shape=vol.shape,
        raw_min=raw_min,
        raw_max=raw_max,
        scalar_path=scalar_path,
        rgb_path=rgb_path,
    )

    print(f"[Done] {subj_id} | shape={vol.shape} | raw range=[{raw_min:.4f}, {raw_max:.4f}]")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    nii_list = sorted(glob.glob(os.path.join(IN_DIR, "*.nii.gz")))
    if not nii_list:
        print(f"No .nii.gz files found in: {IN_DIR}")
        return

    print(f"Found {len(nii_list)} nii.gz files")
    print(f"Output dir: {OUT_DIR}")

    for nii_path in nii_list:
        try:
            process_one_nii(nii_path)
        except Exception as e:
            print(f"[ERROR] {nii_path}: {e}")

    print("All done.")


if __name__ == "__main__":
    main()
