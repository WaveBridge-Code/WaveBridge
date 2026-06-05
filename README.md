# WaveBridge

WaveBridge is a PyTorch/Lightning implementation for 2.5D MRI-to-PET medical image translation with a wavelet-domain diffusion bridge. The training entry point is `main.py`; it combines an NCSN++ style generator, a time-dependent discriminator, Haar wavelet decomposition/reconstruction, and overlap-averaged volume reconstruction for test-time outputs.

This repository contains code only. Datasets, trained checkpoints, generated `.npy`/`.nii.gz` volumes, figures, logs, and paper-result artifacts are intentionally excluded.

## Repository Layout

```text
.
|-- main.py                    # LightningModule and CLI entry point
|-- diffusion.py               # Diffusion bridge forward/posterior/sampling logic
|-- datasets.py                # NumpyDataset and LightningDataModule
|-- utils.py                   # Metrics and visualization helpers
|-- config.yaml                # Sanitized default training/testing config
|-- backbones/                 # Model, wavelet, and CUDA op components
|-- scripts/
|   |-- postprocessing/        # Prediction conversion and visualization utilities
|   `-- analysis/              # Metric/case-selection utilities used for paper analysis
```

## Installation

The original experiments used Python 3.8, CUDA 11.7, PyTorch 1.13.1, and Lightning 2.x.

```bash
conda env create -f environment.yml
conda activate wavebridge
```

If you prefer pip, install PyTorch for your CUDA version first, then install the remaining packages:

```bash
pip install -r requirements.txt
```

The CUDA extensions in `backbones/op/` are compiled on demand by `torch.utils.cpp_extension`. Make sure the local CUDA toolkit and compiler are compatible with your PyTorch build.

## Dataset Format

`NumpyDataset` expects paired `.npy` files under modality and split folders. Each sample should be a multi-channel 2.5D tensor, typically shaped `(C, H, W)` where `C=9` for the default config.

```text
<dataset_root>/
|-- MRI/
|   |-- train/
|   |-- val/
|   `-- test/
`-- PET/
    |-- train/
    |-- val/
    `-- test/
```

Files are paired by subject/slice name. The loader supports names such as `subject_slice_039.npy` and `subject_039.npy`. Inputs should normally be scaled to `[0, 1]`; the dataset maps them to `[-1, 1]` when `data.norm: true`.

## Training

```bash
python main.py fit \
  --config config.yaml \
  --trainer.logger.name wavebridge_mri_pet \
  --data.dataset_dir /path/to/dataset \
  --data.source_modality MRI \
  --data.target_modality PET \
  --data.train_batch_size 4 \
  --data.val_batch_size 4 \
  --trainer.devices [0]
```

For multi-GPU runs, set `--trainer.devices [0,1,...]`. Checkpoints and validation samples are written under the configured CSV logger directory.

## Testing

```bash
python main.py test \
  --config config.yaml \
  --data.dataset_dir /path/to/dataset \
  --data.source_modality MRI \
  --data.target_modality PET \
  --data.test_batch_size 32 \
  --ckpt_path /path/to/checkpoint.ckpt
```

During testing, predictions are saved under the Lightning log directory. The test hook also writes per-slice predictions and overlap-averaged volume arrays.

## Utility Scripts

The scripts in `scripts/postprocessing/` convert prediction arrays to NIfTI files and visualization images. The scripts in `scripts/analysis/` are retained for paper-result bookkeeping, case selection, and metric aggregation. Many of these scripts contain path constants at the top of the file; update them for your local data layout before running.

## Notes

- The repository intentionally ignores data, logs, checkpoints, generated figures, and medical-image volumes.
- `config.yaml` is a portable default. Override paths and run-specific values from the command line.
- The codebase adapts ideas and components from prior diffusion, wavelet, and score-model repositories. See `THIRD_PARTY_NOTICES.md` and file-level headers for details.

## License

The repository keeps the original MIT license file found in the source code bundle. Some included components contain their own file-level notices; retain those notices when modifying or redistributing the code.
