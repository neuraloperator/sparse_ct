import torch
import pytorch_lightning as pl
from src.losses import PSNR, SSIM
from pytorch_lightning.loggers import WandbLogger
from src.utils.compute_metrics import compute_metrics



class VarnetLightningModule(pl.LightningModule):
    def __init__(self, cfg, sampler, sino_reconstructor, image_reconstructor, train_loss, val_test_loss, val_sampler):
        super().__init__()
        self.cfg = cfg
        self.sampler = sampler
        self.val_sampler = val_sampler
        self.test_sampler = val_sampler
        self.sino_reconstructor = sino_reconstructor
        self.image_reconstructor = image_reconstructor
        self.train_loss = train_loss
        self.val_test_loss = val_test_loss

        # don't save large callables
        self.save_hyperparameters(ignore=[
            'sampler', 'val_sampler', 'sino_reconstructor', 'image_reconstructor', 'train_loss', 'val_test_loss'
        ])

        self.psnr = PSNR()
        self.ssim = SSIM()
        
        self.test_zero_psnr = []
        self.test_psnr = []
        self.test_rmse = []
        self.test_ssim = []
        self.radon_transform = None
        
        

    @property
    def name(self):
        return self.cfg.model.name
    
    @property
    def task(self):
        return self.cfg.task
    
    def _calc_loss_by_task(self, recon, image):
        args_dict = {'recon': (recon, image)}
        train_args = args_dict[self.train_loss.task]
        val_test_args = args_dict[self.val_test_loss.task]
        train_loss_value = self.train_loss(*train_args)
        val_test_loss_value = self.val_test_loss(*val_test_args)
        return train_loss_value, val_test_loss_value
    
    def training_step(self, batch, batch_idx):
        image = batch.image
        # sino, theta = radon(image, num_projections=self.cfg.model.num_projections)
        sino, theta = self.radon_transform.radon(image)
        sampled_sino, sampled_theta, sampled_indices = self.sampler(sino, theta)
        sino_recon = self.sino_reconstructor(sampled_sino)
        # raw_image = iradon(sino_recon, sampled_theta)
        raw_image = self.radon_transform.iradon(sino_recon)
        
        sino_recon = self.image_reconstructor(
            sino_recon,
            sampled_sino,
            sampled_indices,
            sampled_theta,
            self.radon_transform
        )

        zero_psnr = self.psnr(raw_image, image)
        image_recon = self.radon_transform.iradon(sino_recon)

        pred_train_loss, train_mse = self._calc_loss_by_task(image_recon, image)

        psnr = self.psnr(image_recon, image)
        mean_ssim = self.ssim(image_recon, image).mean()

        self.log("zero_train_psnr", zero_psnr, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train_psnr", psnr, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train_ssim", mean_ssim, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("train_mse_loss", train_mse, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        return pred_train_loss

    # -----------------------
    # VALIDATION
    # -----------------------
    def validation_step(self, batch, batch_idx):
        image = batch.image
        # sino, theta = radon(image, num_projections=self.cfg.model.num_projections)
        sino, theta = self.radon_transform.radon(image)
        sampled_sino, sampled_theta, sampled_indices = self.sampler(sino, theta)
        sino_recon = self.sino_reconstructor(sampled_sino)
        # raw_image = iradon(sino_recon, sampled_theta)
        raw_image = self.radon_transform.iradon(sino_recon)
        
        sino_recon = self.image_reconstructor(
            sino_recon,
            sampled_sino,
            sampled_indices,
            sampled_theta,
            self.radon_transform
        )
        image_recon = self.radon_transform.iradon(sino_recon)



        zero_psnr = self.psnr(raw_image, image)
        _, val_mse = self._calc_loss_by_task(image_recon, image)
        psnr = self.psnr(image_recon, image)
        mean_ssim = self.ssim(image_recon, image).mean()

        self.log("zero_val_psnr", zero_psnr, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_psnr", psnr, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_ssim", mean_ssim, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log("val_mse_loss", val_mse, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)

        return val_mse

    # -----------------------
    # TEST (robust, distributed-safe)
    # -----------------------
    def on_test_epoch_start(self) -> None:
        """Single-rate setup: allocate K=1 accumulators. Do NOT touch test_sampler."""
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda:0")

        # K = 1 accumulators
        self._sum_zero_psnr    = torch.zeros(1, device=device, dtype=torch.float32)
        self._sum_sq_zero_psnr = torch.zeros(1, device=device, dtype=torch.float32)
        self._sum_psnr         = torch.zeros(1, device=device, dtype=torch.float32)
        self._sum_sq_psnr      = torch.zeros(1, device=device, dtype=torch.float32)
        self._sum_ssim         = torch.zeros(1, device=device, dtype=torch.float32)
        self._sum_sq_ssim      = torch.zeros(1, device=device, dtype=torch.float32)
        self._sum_rmse          = torch.zeros(1, device=device, dtype=torch.float32)
        self._sum_sq_rmse      = torch.zeros(1, device=device, dtype=torch.float32)
        self._test_count       = torch.zeros(1, device=device, dtype=torch.float32)


    def test_single_rate(self, batch, batch_idx):
            idx = 0  # single slot

            image = batch.image
            sino, theta = self.radon_transform.radon(image)
            sampled_sino, sampled_theta, sampled_indices = self.test_sampler(sino, theta)
            sino_recon = self.sino_reconstructor(sampled_sino)
            raw_image = self.radon_transform.iradon(sino_recon)
            sino_recon = self.image_reconstructor(
                sino_recon,
                sampled_sino,
                sampled_indices,
                sampled_theta,
                self.radon_transform
            )
            image_recon = self.radon_transform.iradon(sino_recon)
            
            
            zero_rmse, zero_psnr, zero_ssim = compute_metrics(raw_image, image)
            rmse, psnr, ssim = compute_metrics(image_recon, image)

            self._sum_zero_psnr[idx]    += zero_psnr
            self._sum_sq_zero_psnr[idx] += (zero_psnr ** 2)
            self._sum_psnr[idx]         += psnr
            self._sum_sq_psnr[idx]      += (psnr ** 2)
            self._sum_ssim[idx]         += ssim
            self._sum_sq_ssim[idx]      += (ssim ** 2)
            self._sum_rmse[idx]         += rmse
            self._sum_sq_rmse[idx]      += (rmse ** 2)
            self._test_count[idx]       += image.shape[0]


    def test_step(self, batch, batch_idx):
        # Exactly one rate; script sets self.test_sampler.accn_rate externally.
        self.test_single_rate(batch, batch_idx)


    def on_test_epoch_end(self) -> None:
        """Reduce across ranks and expose results via self.test_results (no logging)."""
        if not hasattr(self, "_test_count"):
            return
        if self._test_count.numel() == 0 or self._test_count.sum().item() == 0.0:
            return

        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        local = torch.stack([
            self._sum_zero_psnr,     # 0
            self._sum_sq_zero_psnr,  # 1
            self._sum_psnr,          # 2
            self._sum_sq_psnr,       # 3
            self._sum_ssim,          # 4
            self._sum_sq_ssim,       # 5
            self._sum_rmse,           # 6
            self._sum_sq_rmse,       # 7
            self._test_count,        # 8
        ], dim=0).to(device)

        try:
            gathered = self.all_gather(local)             # [world, 8, 1] under DDP; [8,1] single
            global_vec = gathered.sum(dim=0) if gathered.dim() == 3 else gathered
        except Exception:
            global_vec = local

        (sum_zero_psnr, sum_sq_zero_psnr,
        sum_psnr, sum_sq_psnr,
        sum_ssim, sum_sq_ssim,
        sum_rmse, sum_sq_rmse, count) = global_vec

        mask = count > 0
        eps = torch.finfo(sum_psnr.dtype).eps

        def mean_of(sum_, cnt):
            out = torch.zeros_like(sum_)
            out[mask] = sum_[mask] / (cnt[mask] + eps)
            return out

        def std_of(sum_, sum_sq_, cnt):
            m = mean_of(sum_, cnt)
            var = torch.zeros_like(sum_)
            var[mask] = (sum_sq_[mask] / (cnt[mask] + eps)) - m[mask] ** 2
            var = torch.clamp(var, min=0.0)
            out = torch.zeros_like(sum_)
            out[mask] = torch.sqrt(var[mask])
            return out

        mean_zero_psnr = mean_of(sum_zero_psnr, count)   # shape [1]
        mean_psnr      = mean_of(sum_psnr,      count)   # shape [1]
        mean_ssim      = mean_of(sum_ssim,      count)   # shape [1]
        mean_rmse      = mean_of(sum_rmse,      count)   # shape [1]

        std_zero_psnr  = std_of(sum_zero_psnr, sum_sq_zero_psnr, count)
        std_psnr       = std_of(sum_psnr,      sum_sq_psnr,      count)
        std_ssim       = std_of(sum_ssim,      sum_sq_ssim,      count)
        std_rmse       = std_of(sum_rmse,      sum_sq_rmse,      count)

        self.test_results = {
            "mean_zero_psnr": mean_zero_psnr,
            "mean_psnr":      mean_psnr,
            "mean_ssim":      mean_ssim,
            "mean_rmse":      mean_rmse,
            "std_zero_psnr":  std_zero_psnr,
            "std_psnr":       std_psnr,
            "std_ssim":       std_ssim,
            "std_rmse":       std_rmse,
            "count":          count,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.trainer.lr)
        return optimizer