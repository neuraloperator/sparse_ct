import os
from typing import Any, Dict, Optional
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
import pathlib
import torch
import numpy as np
import math
import re
from src.samplers import VarnetSparseCTSampler
import wandb

class Test:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        
    def _init_ckpt(self):
        # gather all .ckpt files that start with "best"
        exp_dir = self.cfg.init_exp_dir
        ckpt_paths = [
            p for p in pathlib.Path(exp_dir).iterdir()
            if p.suffix == ".ckpt" and p.stem.startswith("best")
        ]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint files in {exp_dir!r}")

        def version(p: pathlib.Path) -> int:
            # match "best-vN" or just "best"
            m = re.fullmatch(r"best(?:-v(\d+))?", p.stem)
            return int(m.group(1)) if m and m.group(1) else 0

        latest = max(ckpt_paths, key=version)
        if len(ckpt_paths) > 1:
            print(
                f"Warning: multiple checkpoints found, using {latest.name!r}")
        print("Successfully loaded checkpoint from", latest.name)
        return latest
    

    def log_metrics(self, results: Dict[Any, Dict[str, Any]], cc, run_id: Optional[str] = None) -> None:
        """
        Log CT test metrics to Weights & Biases.

        Args:
            results: Mapping {acc_rate: metrics}, where each 'metrics' is:
                {
                    "mean_zero_psnr": float|tensor,
                    "mean_psnr":      float|tensor,
                    "mean_ssim":      float|tensor,
                    "mean_mse":       float|tensor,
                    "rmse":           float|tensor,
                    "std_zero_psnr":  float|tensor,
                    "std_psnr":       float|tensor,
                    "std_ssim":       float|tensor,
                    "count":          int|tensor,
                }
            run_id: Existing W&B run id to resume. If None, a new run is created
                    with a placeholder name. Project is taken from WANDB_PROJECT.
        """
        # --- Optional torch handling (don’t hard-require torch) ---
        
        def _to_float(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                # detach and reduce to scalar if needed
                x = x.detach()
                if x.numel() == 1:
                    return float(x.item())
                # if a tensor slipped in that isn't scalar, take mean as a safe fallback
                return float(x.float().mean().item())
            if isinstance(x, np.generic):
                return float(x)  # numpy scalar
            if isinstance(x, (int, float)):
                return float(x)
            # last resort: try to cast
            try:
                return float(x)
            except Exception:
                return None

        # --- Normalize key types to float for sorting/lookup ---
        norm: Dict[float, Dict[str, Any]] = {}
        for k, v in results.items():
            try:
                r = float(k)
            except Exception:
                # try to coerce strings like "5" cleanly
                r = float(str(k))
            norm[r] = v

        # --- Sort acceleration rates numerically ---
        rates_sorted = sorted(norm.keys())
        # Pretty x-axis: cast whole numbers to int, else keep float
        xs = [int(r) if float(r).is_integer() else float(r) for r in rates_sorted]

        # Helper to collect a list for a given metric over sorted rates
        def L(metric: str):
            vals = []
            for r in rates_sorted:
                vals.append(_to_float(norm[r].get(metric)))
            return vals

        # Collect series
        zpsnr_mu = L("mean_zero_psnr")
        zpsnr_sd = L("std_zero_psnr")
        psnr_mu  = L("mean_psnr")
        psnr_sd  = L("std_psnr")
        ssim_mu  = L("mean_ssim")
        ssim_sd  = L("std_ssim")
        rmse_mu   = L("mean_rmse")
        rmse_sd = L("std_rmse")
        count    = [(_to_float(norm[r].get("count")) if norm[r].get("count") is not None else None) for r in rates_sorted]
        # Ensure integer-like counts are ints for the table
        count = [int(c) if c is not None and not math.isnan(c) else None for c in count]
        print("Results are", results)
        # --- Init or resume W&B run ---
        # If a run is already active, reuse it; otherwise init appropriately.
        # run = wandb.init(mode = "online", name=f"{self.cfg.exp_name}_test")  # placeholder name
            
        # # --- Build and log summary table ---
        # table_cols = [
        #     "accn_rate",
        #     "zero_psnr_mean", "zero_psnr_std",
        #     "psnr_mean", "psnr_std",
        #     "ssim_mean", "ssim_std",
        #     "rmse_mean", "rmse_std",
        #     "count",
        # ]
        # tbl = wandb.Table(columns=table_cols)
        # for i in range(len(xs)):
        #     tbl.add_data(
        #         xs[i],
        #         zpsnr_mu[i], zpsnr_sd[i],
        #         psnr_mu[i],  psnr_sd[i],
        #         ssim_mu[i],  ssim_sd[i],
        #         rmse_mu[i],  rmse_sd[i],
        #         count[i],
        #     )
            
        # print(tbl)
        # run.log({"test/summary_table": tbl})
        # print("Logged table to W&B")
        # run.finish()
        
        wandb_logger = WandbLogger(
            project="YOUR_PROJECT_NAME",
            name=f"{self.cfg.exp_name}_test",
            mode="online",   # same as before
        )

        # --- Build summary table ---
        df = pd.DataFrame({
            "accn_rate": xs,
            "zero_psnr_mean": zpsnr_mu,
            "zero_psnr_std": zpsnr_sd,
            "psnr_mean": psnr_mu,
            "psnr_std": psnr_sd,
            "ssim_mean": ssim_mu,
            "ssim_std": ssim_sd,
            "rmse_mean": rmse_mu,
            "rmse_std": rmse_sd,
            "count": count,
        })

        # --- Log table ---
        wandb_logger.log_table(
            key="test/results_table",
            dataframe=df,
        )

        print("Logged table to W&B")

        # --- Finish run ---
        wandb_logger.experiment.finish()
        
        
    def _to_plain_dict(d):
        import torch
        plain = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()
                if v.numel() == 1:
                    plain[k] = float(v.item())
                else:
                    # vector valued metrics per-rate (K=1 here, but keep robust)
                    plain[k] = [float(x) for x in v.flatten().tolist()]
            else:
                plain[k] = v
        return plain


    def __call__(self, model, data_module):

        accelerations = [10, 20, 40, 80]
        sampler = VarnetSparseCTSampler([1])
        all_results = {}

        for accn_rate in accelerations:
            if hasattr(model, "test_results"):
                delattr(model, "test_results")

            trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                logger=False
            )
            
            _ = trainer.test(model, datamodule=data_module)
            
            res = self._to_plain_dict(model.test_results)
            
            
            def _pick(x):
                if isinstance(x, list):
                    return x[0] if x else None
                return x

            all_results[int(accn_rate)] = {
                "mean_zero_psnr": _pick(res["mean_zero_psnr"]),
                "mean_psnr":      _pick(res["mean_psnr"]),
                "mean_ssim":      _pick(res["mean_ssim"]),
                "mean_rmse":      _pick(res["mean_rmse"]),
                "std_zero_psnr":  _pick(res["std_zero_psnr"]),
                "std_psnr":       _pick(res["std_psnr"]),
                "std_ssim":       _pick(res["std_ssim"]),
                "std_rmse":       _pick(res["std_rmse"]),
                "count":          _pick(res["count"]),
            }
            
            r = all_results[int(accn_rate)]
            print(
                f"[TEST] accn_rate = {int(accn_rate)} | "
                f"PSNR: {r['mean_psnr']:.3f} ± {r['std_psnr']:.3f}, "
                f"SSIM: {r['mean_ssim']:.4f} ± {r['std_ssim']:.4f}, "
                f"RMSE: {r['mean_rmse']:.4f} ± {r['std_rmse']:.4f}, "
                f"Zero-PSNR: {r['mean_zero_psnr']:.3f} ± {r['std_zero_psnr']:.3f} "
                f"(N={r['count']})"
            )
        
        if self.cfg.get("logger"):
            self.log_metrics(all_results)