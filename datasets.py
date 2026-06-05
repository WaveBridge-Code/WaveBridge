from abc import abstractmethod
import os
import yaml
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lightning as L


class BaseDataset(Dataset):
    def __init__(self,
        data_dir,
        target_modality,
        source_modality,
        stage,
        image_size,
        norm=True,
        padding=True
    ):
        self.data_dir = data_dir
        self.target_modality= target_modality
        self.source_modality = source_modality
        self.stage = stage
        self.image_size = image_size
        self.norm = norm
        self.padding = padding
        self.original_shape = None

    @abstractmethod
    def _load_data(self, contrast):
        pass

    def _pad_data(self, data):
        """ Pad data to image_size x image_size """
        H, W = data.shape[-2:]

        pad_top = (self.image_size - H) // 2
        pad_bottom = self.image_size - H - pad_top
        pad_left = (self.image_size - W) // 2
        pad_right = self.image_size - W - pad_left

        return np.pad(data, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))

    def _normalize(self, data):
        return (data - 0.5) / 0.5


class NumpyDataset(Dataset):
    def __init__(
        self,
        data_dir,
        target_modality,
        source_modality,
        stage,
        image_size,
        norm=True,
        padding=True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.target_modality = target_modality
        self.source_modality = source_modality
        self.stage = stage
        self.image_size = image_size
        self.norm = norm
        self.padding = padding

        # 获取 target / source 文件字典（key -> path�?        target_dict = self._get_file_dict(target_modality)
        source_dict = self._get_file_dict(source_modality)

        # 取交集键，确保严格一一对应
        common_keys = sorted(list(set(target_dict.keys()) & set(source_dict.keys())))
        if len(common_keys) == 0:
            raise RuntimeError(f"No matching files found between {target_modality} and {source_modality} for stage {stage} in {data_dir}")

        # 构造对应的文件列表
        self.target_files = [target_dict[k] for k in common_keys]
        self.source_files = [source_dict[k] for k in common_keys]
        self.slice_filenames = [os.path.basename(p) for p in self.source_files]

        # subject ids (optional yaml)
        self.subject_ids = self._load_subject_ids('subject_ids.yaml')

        # debug / sanity print
        print(f"[NumpyDataset] {stage} - modality pair ({source_modality} <- {target_modality}): found {len(self.target_files)} paired samples.")

    def _get_file_dict(self, contrast):
        """
        递归搜索 contrast/stage 下的所�?.npy 文件，返�?dict: key -> fullpath
        key 格式: '{patient_id}__{slice_idx:03d}'
        允许文件名格式：'{patient_id}_slice_000.npy' �?'{patient_id}_000.npy' �?        """
        data_dir = os.path.join(self.data_dir, contrast, self.stage)
        # 递归查找
        pattern = os.path.join(data_dir, "**", "*.npy")
        files = glob.glob(pattern, recursive=True)

        def make_key(fname: str) -> str:
            basename = os.path.basename(fname)

            # 优先按你现在的规则：..._slice_XXX.npy
            if "_slice_" in basename:
                subject_id, slice_part = basename.split("_slice_", 1)
                slice_str = os.path.splitext(slice_part)[0]
                if not slice_str.isdigit():
                    raise ValueError(
                        f"File name '{basename}' has '_slice_' but slice index '{slice_str}' is not integer."
                    )
                slice_idx = int(slice_str)
                patient_id = subject_id  # 这里会保�?_2,_3 等时间点后缀
            else:
                # 兼容旧命名：xxx_xxx_039.npy 这种情况
                parts = basename.split('_')
                last = parts[-1]
                last_noext = os.path.splitext(last)[0]
                if last_noext.isdigit():
                    slice_idx = int(last_noext)
                    patient_id = "_".join(parts[:-1])
                else:
                    # 实在解析不出 slice，就当成一个整体，slice_idx=0
                    patient_id = os.path.splitext(basename)[0]
                    slice_idx = 0

            return f"{patient_id}__{slice_idx:03d}"

        file_dict = {}
        for f in files:
            k = make_key(f)
            if k in file_dict:
                # 如果有重�?key，可以在这里打印一下调试信息，看是不是数据本身重复�?                # print(f"[Warning] duplicate key {k} for file {f} (existing {file_dict[k]})")
                continue
            file_dict[k] = f

        return file_dict

    def _load_subject_ids(self, filename):
        subject_ids_path = os.path.join(self.data_dir, filename)
        if os.path.exists(subject_ids_path):
            with open(subject_ids_path, 'r') as f:
                subject_ids = np.array(yaml.load(f, Loader=yaml.FullLoader))
        else:
            subject_ids = None
        return subject_ids

    def _load_and_process(self, file_path):
        """加载单个 .npy 文件，并执行 padding / normalize"""
        data = np.load(file_path).astype(np.float32)

        # padding (data �?2D: H x W)
        if self.padding:
            data = self._pad_data(data)

        # normalize
        if self.norm:
            data = self._normalize(data)

        return data

    def _pad_data(self, data):
        """
        多通道 padding�?        输入: (C,H,W)
        输出: (C,newH,newW)
        """
        C, H, W = data.shape
        pad_top = (self.image_size - H) // 2
        pad_bottom = self.image_size - H - pad_top
        pad_left = (self.image_size - W) // 2
        pad_right = self.image_size - W - pad_left
        return np.pad(
            data,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant"
        )

    def _normalize(self, data):
        # 如果数据原本�?[0,1]�?x-0.5)/0.5 映射�?[-1,1]
        return (data - 0.5) / 0.5

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, i):
        target = self._load_and_process(self.target_files[i])
        source = self._load_and_process(self.source_files[i])
        filename = self.slice_filenames[i]
        return target, source, filename, i

class DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_dir,
        source_modality,
        target_modality,
        dataset_class,
        image_size,
        padding,
        norm,
        train_batch_size=1,
        val_batch_size=1,
        test_batch_size=1,
        num_workers=1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_dir = dataset_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.source_modality = source_modality
        self.target_modality = target_modality
        self.image_size = image_size
        self.padding = padding
        self.norm = norm
        self.num_workers = num_workers

        self.dataset_class = globals()[dataset_class]

    def setup(self, stage: str) -> None:
        target_modality = self.target_modality
        source_modality = self.source_modality

        if stage == "fit":
            self.train_dataset = self.dataset_class(
                target_modality=target_modality,
                source_modality=source_modality,
                data_dir=self.dataset_dir,
                stage='train',
                image_size=self.image_size,
                padding=self.padding,
                norm=self.norm
            )

            self.val_dataset = self.dataset_class(
                target_modality=target_modality,
                source_modality=source_modality,
                data_dir=self.dataset_dir,
                stage='val',
                image_size=self.image_size,
                padding=self.padding,
                norm=self.norm
            )

        if stage == "test":
            self.test_dataset = self.dataset_class(
                target_modality=target_modality,
                source_modality=source_modality,
                data_dir=self.dataset_dir,
                stage='test',
                image_size=self.image_size,
                padding=self.padding,
                norm=self.norm
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False
        )
