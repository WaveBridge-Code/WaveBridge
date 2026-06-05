import os
import piq
import numpy as np
import torch
import wandb
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import lightning as L
from lightning.pytorch.cli import LightningCLI
from diffusion import DiffusionBridge
from backbones.ncsnpp import NCSNpp
from backbones.DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D
from pytorch_wavelets import DWTForward, DWTInverse
from backbones.discriminator import Discriminator_large
from datasets import DataModule
from utils import compute_metrics, save_image_pair, save_preds, save_eval_images
from pytorch_msssim import ssim
from torch import distributed as dist



class BridgeRunner(L.LightningModule):
    def __init__(
        self,
        generator_params,
        discriminator_params,
        diffusion_params,
        lr_g,
        lr_d,
        disc_grad_penalty_freq,
        disc_grad_penalty_weight,
        lambda_rec_loss,
        lambda_content_loss,
        lambda_hf,
        optim_betas,
        eval_mask,
        eval_subject,
        use_pytorch_wavelet,
        current_resolution,
        use_wandb,
        input_channels: int = 9,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.automatic_optimization = False
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.disc_grad_penalty_freq = disc_grad_penalty_freq
        self.disc_grad_penalty_weight = disc_grad_penalty_weight
        self.lambda_rec_loss = lambda_rec_loss
        self.lambda_content_loss = lambda_content_loss
        self.lambda_hf = lambda_hf
        self.optim_betas = optim_betas
        self.eval_mask = eval_mask
        self.eval_subject = eval_subject
        self.current_resolution = current_resolution
        self.n_steps = diffusion_params['n_steps']
        self.n_recursions = diffusion_params['n_recursions']
        self.use_pytorch_wavelet = use_pytorch_wavelet
        self.use_wandb = use_wandb
        self.input_channels = input_channels
        # “中心切片”的索引（比�?C=9 时为 4�?        self.center_index = input_channels // 2
        self.cri_fea = nn.L1Loss().to(self.device)
        # if self.cri_fea:  # load VGG perceptual loss
        #     self.netF = VGGLoss(device=self.device)

        # Networks
        self.generator = NCSNpp(**generator_params)
        self.discriminator = Discriminator_large(**discriminator_params)

        # Configure diffusion
        self.diffusion = DiffusionBridge(**diffusion_params)
        if not self.use_pytorch_wavelet:
            self.dwt = DWT_2D("haar")
            self.iwt = IDWT_2D("haar")
        else:
            self.dwt = DWTForward(J=1, mode='zero', wave='haar').cuda()
            self.iwt = DWTInverse(mode='zero', wave='haar').cuda()
    def on_fit_start(self):
        if self.use_wandb and self.global_rank == 0 and os.environ.get("WANDB_INITED") is None:
            wandb.init(
                project="selfrdb",
                name=f"train_{wandb.util.generate_id()}",
                config=self.hparams,
                mode='offline',
                settings=wandb.Settings(init_timeout=300)
            )
            os.environ["WANDB_INITED"] = "1"  # 设定环境变量，防止重复初始化

    def training_step(self, batch):
        torch.set_grad_enabled(True)
        torch.use_deterministic_algorithms(False)
        x0, y, filename, _ = batch
        optimizer_g, optimizer_d = self.optimizers()
        scheduler_g, scheduler_d = self.lr_schedulers()

        num_levels = int(np.log2( 256 // self.current_resolution))
        # Part 1: Train discriminator训练判别�?        self.toggle_optimizer(optimizer_d)
        if not self.use_pytorch_wavelet:
            for i in range(num_levels):
                xll, xlh, xhl, xhh = self.dwt(x0)
                yll, ylh, yhl, yhh = self.dwt(y)
        else:
            xll, xh = self.dwt(x0)
            yll, yh = self.dwt(y)
            xlh, xhl, xhh = torch.unbind(xh[0], dim=2)
            ylh, yhl, yhh = torch.unbind(yh[0], dim=2)
        real_data = torch.cat([xll, xlh, xhl, xhh], dim=1)  # [b, 4*input_channels, h, w]
        y_data = torch.cat([yll, ylh, yhl, yhh], dim=1)  # [b, 4*input_channels, h, w]
        # Part 1.a: Train discriminator with real data
        # Sample a time step
        t = torch.randint(1, self.n_steps+1, (real_data.shape[0],)).to(x0.device)

        # Sample x_{t-1} and x_t via forward process 前向过程加噪
        x_tm1 = self.diffusion.q_sample(t - 1, real_data, y_data)
        x_t = self.diffusion.q_sample(t, real_data, y_data)
        x_t.requires_grad = True
        # Perform real data prediction 执行真实数据预测
        disc_out = self.discriminator(x_tm1, x_t, t)
        real_loss = self.adversarial_loss(disc_out, is_real=True)
        disc_real_acc = (disc_out > 0).float().mean()

        # Compute gradient penalty计算梯度惩罚
        if self.global_step % self.disc_grad_penalty_freq == 0:
            grads = torch.autograd.grad(outputs=disc_out.sum(), inputs=x_t, create_graph=True)[0]
            grad_penalty = (grads.view(grads.size(0), -1).norm(2, dim=1) ** 2).mean()
            grad_penalty = grad_penalty * self.disc_grad_penalty_weight
            real_loss += grad_penalty

        # Part 1.b: Train discriminator with fake data
        # Perform recursive x0 prediction
        x0_r = torch.zeros_like(x_t)
        for _ in range(self.n_recursions):
            x0_r = self.generator(torch.cat((x_t.detach(), y_data), axis=1), t, x_r=x0_r)
        x0_pred = x0_r
        # Posterior sampling q(x_{t-1} | x_t, y, x0_pred)
        x_tm1_pred = self.diffusion.q_posterior(t, x_t, x0_pred, y_data)
        assert x_tm1_pred.shape == x_t.shape, f"x_tm1_pred shape {x_tm1_pred.shape} does not match x_t shape {x_t.shape}"
        # Perform fake data prediction
        disc_out = self.discriminator(x_tm1_pred, x_t, t)
        fake_loss = self.adversarial_loss(disc_out, is_real=False)
        disc_fake_acc = (disc_out < 0).float().mean()

        # Compute discriminator accuracy
        d_acc = (disc_real_acc + disc_fake_acc) / 2

        # Compute total loss
        d_loss = real_loss + fake_loss

        # Perform backprop
        self.manual_backward(d_loss, retain_graph=True)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        # Part 2: Train generator
        self.toggle_optimizer(optimizer_g)

        # Sample a time step
        t = torch.randint(1, self.n_steps + 1, (real_data.shape[0],)).to(x0.device)

        # Get x_t via forward process 前向加噪
        x_t = self.diffusion.q_sample(t, real_data, y_data)

        # assert x_t.shape == x0.shape, f"IWT output shape {x_t.shape} doesn't match original x0 shape {x0.shape}"
        # Perform recursive x0 prediction
        x0_r = torch.zeros_like(x_t)
        for _ in range(self.n_recursions):
            x0_r = self.generator(torch.cat((x_t.detach(), y_data), axis=1), t, x_r=x0_r)

        x0_pred = x0_r
        C = self.input_channels
        # x0_pred: (B, 4*C, H, W) = (B, 36, H, W)
        xll_pred = x0_pred[:, 0:C, :, :]  # (B, 9, H, W)
        xlh_pred = x0_pred[:, C:2 * C, :, :]
        xhl_pred = x0_pred[:, 2 * C:3 * C, :, :]
        xhh_pred = x0_pred[:, 3 * C:4 * C, :, :]

        x0_pred_image = self.iwt(xll_pred, xlh_pred, xhl_pred, xhh_pred)
        hf_sparsity_loss = F.l1_loss(xhh_pred, torch.zeros_like(xhh_pred))

        x0_pred_image_norm = ((x0_pred_image + 1) / 2).clamp(0, 1)
        x0_norm = ((x0 + 1) / 2).clamp(0, 1)
        # Posterior sampling q(x_{t-1} | x_t, y, x0_pred)
        x_tm1_pred = self.diffusion.q_posterior(t, x_t, x0_pred, y_data)

        # Compute reconstruction loss
        rec_loss = F.l1_loss(x0_pred, real_data, reduction="sum")
        # 计算内容损失�? - SSIM
        content_loss = 1 - ssim(x0_pred_image_norm, x0_norm, data_range=1.0, size_average=True)

        # Ensure x_tm1_pred matches x_t in shape before feeding into discriminator
        assert x_tm1_pred.shape == x_t.shape, f"x_tm1_pred shape {x_tm1_pred.shape} does not match x_t shape {x_t.shape}"

        # Compute adversarial loss
        adv_loss = self.adversarial_loss(
            self.discriminator(x_tm1_pred, x_t, t), is_real=True)
        
        # Compute total loss and perform backprop
        g_loss = self.lambda_rec_loss*rec_loss + adv_loss + self.lambda_content_loss*content_loss + self.lambda_hf * hf_sparsity_loss

        # Perform backprop
        self.manual_backward(g_loss)

        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        # Take lr scheduler step
        scheduler_g.step()
        scheduler_d.step()

        if self.use_wandb and self.global_rank == 0:
            # 在原有的日志记录后添�?wandb 记录
            wandb.log({
                "train/d_loss": d_loss,
                "train/g_loss_rec": rec_loss,
                "train/g_loss_content": content_loss,
                "train/g_loss_adv": adv_loss,
                "train/g_loss_total": g_loss,
            }, step=self.global_step)
        # Log losses
        self.log("d_loss", d_loss, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log("g/rec", rec_loss, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log("g/content", self.lambda_content_loss*content_loss, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log("g/hf", self.lambda_hf * hf_sparsity_loss, on_epoch=True, prog_bar=True, sync_dist=True,
                 rank_zero_only=True)
        self.log("g/adv", adv_loss, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log("g/total", g_loss, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)

    def validation_step(self, batch, batch_idx):
        x0, y, filename, _ = batch
        center = self.center_index
        x0_center = x0[:, center:center + 1, :, :]  # (B,1,H,W)
        # Predict x0
        x0_pred = self.diffusion.sample_x0(y, self.generator)
        C = self.input_channels
        # x0_pred: (B, 4*C, H, W) = (B, 36, H, W)
        xll_pred = x0_pred[:, 0:C, :, :]  # (B, 9, H, W)
        xlh_pred = x0_pred[:, C:2 * C, :, :]
        xhl_pred = x0_pred[:, 2 * C:3 * C, :, :]
        xhh_pred = x0_pred[:, 3 * C:4 * C, :, :]
        if not self.use_pytorch_wavelet:
            xll_gt, xlh_gt, xhl_gt, xhh_gt = self.dwt(x0)
        else:
            xll_gt, xh_gt = self.dwt(x0)
            xlh_gt, xhl_gt, xhh_gt = torch.unbind(xh_gt[0], dim=2)
        real_data = torch.cat([xll_gt, xlh_gt, xhl_gt, xhh_gt], dim=1)  # (B,36,h,w)
        pred_data = torch.cat([xll_pred, xlh_pred, xhl_pred, xhh_pred], dim=1)
        freq_loss = F.mse_loss(pred_data, real_data)
        pred_imgs = []
        for i in range(C):
            LL_i = xll_pred[:, i:i + 1]
            LH_i = xlh_pred[:, i:i + 1]
            HL_i = xhl_pred[:, i:i + 1]
            HH_i = xhh_pred[:, i:i + 1]
            img_i = self.iwt(LL_i, LH_i, HL_i, HH_i)
            pred_imgs.append(img_i)

        pred_imgs = torch.cat(pred_imgs, dim=1)  # (B,9,H,W)
        center_pred_img = pred_imgs[:, self.center_index:self.center_index + 1]

        # 将频域还原为图像�?        if not self.use_pytorch_wavelet:
            x0_pred_image = self.iwt(
                xll_pred,  # LL
                xlh_pred,  # LH
                xhl_pred,  # HL
                xhh_pred  # HH
            )
        else:
            xh = torch.stack(
                (xlh_pred, xhl_pred, xhh_pred), dim=2
            )  # (B,1,3,H,W)

            x0_pred_image = self.iwt(
                (xll_pred, [xh])
            )
        # ===== 中心切片（图像域�?====
        x0_pred_center_img = x0_pred_image[:, self.center_index:self.center_index + 1]

        loss = F.mse_loss(x0_pred_image, x0)
        # === 对指标计算时使用的副本归一�?===
        x0_norm = ((x0_center + 1) / 2).clamp(0, 1)
        x0_pred_image_norm = ((x0_pred_center_img + 1) / 2).clamp(0, 1)
        metrics = compute_metrics(x0_norm, x0_pred_image_norm)
        psnr_mean = metrics["psnr_mean"].mean()
        ssim_mean = metrics["ssim_mean"].mean()
        if self.use_wandb and self.global_rank == 0:# 记录验证指标
            wandb.log({
                "val/loss": loss,
                "val/freq_loss": freq_loss,
                "val/psnr": psnr_mean,
                "val/ssim": ssim_mean
            }, step=self.global_step)
        # Log metrics
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log("val_psnr", psnr_mean, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log("val_ssim", ssim_mean, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True)

        # Log sample images
        if batch_idx == 0 and self.global_rank == 0:
            path = os.path.join(self.logger.log_dir, "val_samples", f"epoch_{self.current_epoch}.png")
            save_image_pair(x0_center, center_pred_img, path)
            # 保存 range 信息
            x0_np = x0_center.detach().cpu().numpy()
            x0_pred_np = center_pred_img.detach().cpu().numpy()
            min_x0, max_x0 = x0_np.min(), x0_np.max()
            min_pred, max_pred = x0_pred_np.min(), x0_pred_np.max()

            range_log = (
                f"Epoch {self.current_epoch}\n"
                f"x0       range: ({min_x0:.6f}, {max_x0:.6f})\n"
                f"x0_pred  range: ({min_pred:.6f}, {max_pred:.6f})\n"
            )
            range_log_path = os.path.join(self.logger.log_dir, "val_samples",
                                          f"epoch_{self.current_epoch}_range_info.txt")
            with open(range_log_path, "w") as f:
                f.write(range_log)

    def _parse_volume_and_slice(self, filename):
        base = os.path.basename(str(filename))
        name_noext = os.path.splitext(base)[0]

        if "_slice_" in name_noext:
            volume_id, slice_str = name_noext.rsplit("_slice_", 1)
            return volume_id, int(slice_str)

        parts = name_noext.split("_")
        if parts[-1].isdigit():
            return "_".join(parts[:-1]), int(parts[-1])

        raise ValueError(f"Cannot parse slice index from filename: {filename}")

    def _save_overlap_averaged_predictions(self):
        test_dir = os.path.join(self.logger.log_dir, "test_samples")
        slice_save_dir = os.path.join(test_dir, "pred_slices_overlap_avg")
        volume_save_dir = os.path.join(test_dir, "pred_volumes_overlap_avg")
        os.makedirs(slice_save_dir, exist_ok=True)
        os.makedirs(volume_save_dir, exist_ok=True)

        center = self.center_index
        C = self.input_channels
        dataset = self.trainer.datamodule.test_dataset

        # 线性权重：中心切片权重最大，越靠近边缘权重越�?        linear_weights = np.array(
            [center + 1 - abs(k - center) for k in range(C)],
            dtype=np.float32
        )

        print(f"Using linear overlap weights: {linear_weights.tolist()}")

        valid_slices = {}
        for path in dataset.source_files:
            volume_id, slice_idx = self._parse_volume_and_slice(path)
            valid_slices.setdefault(volume_id, set()).add(slice_idx)

        # 先按窗口去重，避�?DDP sampler padding 导致重复窗口被重复平�?        window_preds = {}
        for gathered in self.recon_objects:
            for obj in gathered:
                preds = obj["pred"]  # (B, 15, H, W)
                filenames = obj["filename"]

                for b, fname in enumerate(filenames):
                    volume_id, center_slice = self._parse_volume_and_slice(fname)
                    key = (volume_id, center_slice)

                    if key not in window_preds:
                        window_preds[key] = preds[b]

        volume_sums = {}
        volume_weights = {}

        for (volume_id, center_slice), pred_15 in window_preds.items():
            volume_sums.setdefault(volume_id, {})
            volume_weights.setdefault(volume_id, {})

            for k in range(C):
                real_slice_idx = center_slice + k - center

                # 避免边界处产生不存在�?slice
                if real_slice_idx not in valid_slices.get(volume_id, set()):
                    continue

                weight = linear_weights[k]

                if real_slice_idx not in volume_sums[volume_id]:
                    volume_sums[volume_id][real_slice_idx] = np.zeros_like(
                        pred_15[k], dtype=np.float32
                    )
                    volume_weights[volume_id][real_slice_idx] = 0.0

                volume_sums[volume_id][real_slice_idx] += (
                        pred_15[k].astype(np.float32) * weight
                )
                volume_weights[volume_id][real_slice_idx] += weight

        for volume_id in sorted(volume_sums.keys()):
            slice_indices = sorted(volume_sums[volume_id].keys())
            per_volume_slice_dir = os.path.join(slice_save_dir, volume_id)
            os.makedirs(per_volume_slice_dir, exist_ok=True)

            avg_slices = []
            for slice_idx in slice_indices:
                weight_sum = volume_weights[volume_id][slice_idx]
                if weight_sum <= 0:
                    raise ValueError(f"Zero weight for {volume_id}, slice {slice_idx}")

                avg_slice = volume_sums[volume_id][slice_idx] / weight_sum
                avg_slice = avg_slice.astype(np.float32)
                avg_slices.append(avg_slice)

                np.save(
                    os.path.join(per_volume_slice_dir, f"{volume_id}_slice_{slice_idx:03d}.npy"),
                    avg_slice
                )
            volume_h_w_z = np.stack(avg_slices, axis=-1).astype(np.float32)
            np.save(os.path.join(volume_save_dir, f"{volume_id}.npy"), volume_h_w_z)

            print(
                f"Saved overlap-averaged volume: {volume_id}, "
                f"shape={volume_h_w_z.shape}, slices={len(slice_indices)}"
            )

        print(f"Saved overlap-averaged slices to {slice_save_dir}")
        print(f"Saved overlap-averaged volumes to {volume_save_dir}")
    def on_fit_end(self):
        if self.global_rank == 0:
            wandb.finish()

    def on_test_start(self):
        self.test_samples = []
        self.recon_objects = []
        self.psnrs = []
        self.ssims = []
        self.mask = None
        self.subject_ids = None

        # Load mask for evaluation
        if self.eval_mask:
            self.mask = self.trainer.datamodule.test_dataset._load_data('mask')

        # Load subject ids for evaluation
        if self.eval_subject:
            self.subject_ids = self.trainer.datamodule.test_dataset.subject_ids

    def test_step(self, batch, batch_idx):
        x0, y, filename, slice_idx = batch
        center = self.center_index
        x0_center = x0[:, center:center + 1, :, :]  # (B,1,H,W)
        # Predict x0
        x0_pred = self.diffusion.sample_x0(y, self.generator)
        C = self.input_channels
        # x0_pred: (B, 4*C, H, W) = (B, 36, H, W)
        xll_pred = x0_pred[:, 0:C, :, :]  # (B, 9, H, W)
        xlh_pred = x0_pred[:, C:2 * C, :, :]
        xhl_pred = x0_pred[:, 2 * C:3 * C, :, :]
        xhh_pred = x0_pred[:, 3 * C:4 * C, :, :]

        # 将频域还原为图像�?        if not self.use_pytorch_wavelet:
            x0_pred_image = self.iwt(
                xll_pred,  # LL
                xlh_pred,  # LH
                xhl_pred,  # HL
                xhh_pred  # HH
            )
        else:
            xh = torch.stack(
                (xlh_pred, xhl_pred, xhh_pred), dim=2
            )  # (B,1,3,H,W)

            x0_pred_image = self.iwt(
                (xll_pred, [xh])
            )
        center_pred_img = x0_pred_image[:, center:center + 1]
        # Gather pred images across all ranks
        all_pred = self.all_gather(center_pred_img)
        slice_indices = self.all_gather(slice_idx)
        
        if self.global_rank == 0:
            h, w = x0_center.shape[-2:]
            self.test_samples.extend(list(zip(
                slice_indices.flatten().tolist(),
                all_pred.reshape(-1, h, w).cpu().numpy())))

        local_indices = slice_idx.detach().cpu().numpy()
        local_preds = x0_pred_image.detach().cpu().numpy().astype(np.float32)
        local_obj = {
            "slice_idx": local_indices,
            "pred": local_preds,  # (B, 15, H, W)，用于重叠平均重�?            "filename": list(filename)
        }

        gathered = [None for _ in range(dist.get_world_size())] if dist.is_available() and dist.is_initialized() else [
            local_obj]
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(gathered, local_obj)

        if self.global_rank == 0:
            self.recon_objects.append(gathered)
        
    def on_test_end(self):
        # Save predicted images
        if self.global_rank == 0:
            os.makedirs(os.path.join(self.logger.log_dir, "test_samples"), exist_ok=True)
            # Sort samples by slice index
            self.test_samples.sort(key=lambda x: x[0])
            
            # Extract pred images
            pred = np.array([x[1] for x in self.test_samples])
            slice_indices = np.array([x[0] for x in self.test_samples])

            # Remove repeated slices that can occur in multi-GPU setting
            _, locs = np.unique(slice_indices, return_index=True)
            pred = pred[locs]
            slice_indices = slice_indices[locs]
            path = os.path.join(self.logger.log_dir, "test_samples", "pred.npy")
            save_preds(pred, path)

            # Get source and target images
            dataset = self.trainer.datamodule.test_dataset
            target_all = np.stack([dataset._load_and_process(dataset.target_files[i]) for i in range(len(dataset))])
            source_all = np.stack([dataset._load_and_process(dataset.source_files[i]) for i in range(len(dataset))])
            center = target_all.shape[1] // 2
            target = target_all[:, center:center + 1]
            source = source_all[:, center:center + 1]

            # 创建输出目录
            slice_save_dir = os.path.join(self.logger.log_dir, "test_samples", "pred_slices")
            os.makedirs(slice_save_dir, exist_ok=True)

            assert pred.shape[0] == len(dataset), \
                f"pred 数量 {pred.shape[0]} �?dataset 长度 {len(dataset)} 不一致！"

            for i in range(len(dataset)):
                # pred[i]: 对应 test_dataset 中第 i 个样本的中心切片 (H, W)
                p = pred[i]

                # 对应的原始文件名�? 通道 npy�?                orig_path = dataset.source_files[i]
                base = os.path.basename(orig_path)  # e.g. 135_S_6586_slice_039.npy
                name_noext, ext = os.path.splitext(base)
                if ext == "":
                    ext = ".npy"

                if "_slice_" in name_noext:
                    # 例如 135_S_6586_2_slice_039.npy -> 保持原样
                    save_name = name_noext + ext
                else:
                    # 兼容旧格式：�?135_S_6586_039.npy
                    parts = name_noext.split("_")
                    if parts[-1].isdigit():
                        patient_id = "_".join(parts[:-1])
                        slice_no = int(parts[-1])
                    else:
                        patient_id = name_noext
                        slice_no = i  # 退化情况，�?i �?slice �?                    save_name = f"{patient_id}_slice_{slice_no:03d}{ext}"

                # p 本身�?(H,W)，无需�?squeeze
                np.save(os.path.join(slice_save_dir, save_name), p.astype(np.float32))

            print(f"�?Saved all slices to {slice_save_dir}")
            self._save_overlap_averaged_predictions()


            # Compute metrics and save report计算指标并保存报�?            metrics = compute_metrics(
                gt_images=target,
                pred_images=pred,
                mask=self.mask,
                subject_ids=self.subject_ids,
                report_path=os.path.join(self.logger.log_dir, "test_samples", "report.txt"),
            )

            # Print metrics
            print(f"PSNR: {metrics['psnr_mean']:.2f} ± {metrics['psnr_std']:.2f}")
            print(f"SSIM: {metrics['ssim_mean']:.2f} ± {metrics['ssim_std']:.2f}")
            print(f"LPIPS: {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")
            print(f"MAE: {metrics['mae_mean']:.6f} ± {metrics['mae_std']:.6f}")
            # 根据 slice_indices 排序保持一�?            sort_idx = np.argsort(slice_indices)
            pred_sorted = pred[sort_idx]
            target_sorted = target[sort_idx]
            source_sorted = source[sort_idx]
            # Save sample images  选择 10 张样本进行可视化
            indices = np.random.choice(len(dataset), 20)
            path = os.path.join(self.logger.log_dir, "test_samples")
            save_eval_images(
                source_images=source_sorted[indices],
                target_images=target_sorted[indices],
                pred_images=pred_sorted[indices],
                psnrs=metrics["psnrs"][sort_idx][indices],
                ssims=metrics["ssims"][sort_idx][indices],
                save_path=os.path.join(self.logger.log_dir, "test_samples")
            )


            eval_path = os.path.join(self.logger.log_dir, "test_samples", "pred.npy")
            print(f"Saved batch-aligned pred.npy to {eval_path}")


    def adversarial_loss(self, pred, is_real):
        loss = F.softplus(-pred) if is_real else F.softplus(pred)
        return loss.mean()
    
    def configure_optimizers(self):
        optimizer_g = Adam(list(self.generator.parameters()), lr=self.lr_g, betas=self.optim_betas)
        optimizer_d = Adam(self.discriminator.parameters(), lr=self.lr_d, betas=self.optim_betas)
        
        # Learning rate schedulers
        scheduler_g = CosineAnnealingLR(optimizer_g, T_max=self.trainer.max_epochs, eta_min=1e-5)
        scheduler_d = CosineAnnealingLR(optimizer_d, T_max=self.trainer.max_epochs, eta_min=1e-5)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]


class _LightningCLI(LightningCLI):
    def instantiate_classes(self):
        # Log to checkpoint directory when testing
        if 'test' in self.parser.args and 'CSVLogger' in self.config.test.trainer.logger[0].class_path:
            exp_dir = os.path.dirname(os.path.dirname(self.config.test.ckpt_path))
            logger = self.config.test.trainer.logger[0]
            logger.init_args.save_dir = os.path.dirname(exp_dir)
            logger.init_args.name = os.path.basename(exp_dir)
            logger.init_args.version = "test"

        super().instantiate_classes()

    def before_fit(self):
        """�?在训练开始前执行，用于调试或检查数�?""
        datamodule = self.datamodule
        datamodule.setup("fit")

        print("\n==============================")
        print("🔍 检查前10�?target/source 文件是否一一对应")
        dataset = datamodule.train_dataset

        # 检查前10个文�?        for t, s in zip(dataset.target_files[:10], dataset.source_files[:10]):
            print(f"{os.path.basename(t)} <---> {os.path.basename(s)}")
        print("==============================\n")

torch.set_float32_matmul_precision("medium")

# 配置 cuDNN 以优化计�?torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # �?cuDNN 选择最优算�?torch.backends.cudnn.deterministic = False  # 避免固定计算模式

def cli_main():
    cli = _LightningCLI(
        BridgeRunner,
        DataModule,
        save_config_callback=None,
        parser_kwargs={"parser_mode": "omegaconf"},
    )



if __name__ == "__main__":
    cli_main()
