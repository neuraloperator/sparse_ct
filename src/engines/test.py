import os
import json
import pathlib
import re
import math
from datetime import datetime
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import numpy as np

# If you still need this elsewhere in the file, keep it; otherwise remove.
# from src.samplers import VarnetSparseCTSampler


class Test:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    # ---------- small cfg helpers ----------
    def _cfg_get(self, key: str, default=None):
        # works for dict-like (OmegaConf supports .get) and attribute-style cfgs
        if hasattr(self.cfg, "get"):
            v = self.cfg.get(key, default)
            return default if v is None else v
        return getattr(self.cfg, key, default)

    def _cfg_exp_name(self) -> str:
        return str(self._cfg_get("exp_name", "experiment"))

    # ---------- checkpoint helper ----------
    def _init_ckpt(self):
        exp_dir = self._cfg_get("init_exp_dir")
        if exp_dir is None:
            raise ValueError("cfg.init_exp_dir is missing")

        ckpt_paths = [
            p for p in pathlib.Path(exp_dir).iterdir()
            if p.suffix == ".ckpt" and p.stem.startswith("best")
        ]
        if not ckpt_paths:
            raise FileNotFoundError(f"No checkpoint files in {exp_dir!r}")

        def version(p: pathlib.Path) -> int:
            m = re.fullmatch(r"best(?:-v(\d+))?", p.stem)
            return int(m.group(1)) if m and m.group(1) else 0

        latest = max(ckpt_paths, key=version)
        if len(ckpt_paths) > 1:
            print(f"Warning: multiple checkpoints found, using {latest.name!r}")
        print("Successfully loaded checkpoint from", latest.name)
        return latest

    # ---------- metrics extraction ----------
    def _to_plain_dict(self, d):
        plain = {}
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu()
                if v.numel() == 1:
                    plain[k] = float(v.item())
                else:
                    plain[k] = [float(x) for x in v.flatten().tolist()]
            else:
                plain[k] = v
        return plain

    def _pick(self, x):
        # your metrics sometimes come as list[tensor/float] with len=1
        if isinstance(x, list):
            return x[0] if x else None
        return x

    # ---------- saving + printing ----------
    def _ensure_results_dir(self) -> str:
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _format_pm(self, mu, sd, mu_fmt="{:.3f}", sd_fmt="{:.3f}"):
        if mu is None or sd is None:
            return "NA"
        return f"{mu_fmt.format(mu)} ± {sd_fmt.format(sd)}"

    def save_results(self, all_results: Dict[int, Dict[str, Any]]) -> None:
        out_dir = self._ensure_results_dir()
        exp_name = self._cfg_exp_name()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Build rows (REMOVED zero-psnr fields)
        rows = []
        for rate in sorted(all_results.keys()):
            r = all_results[rate]
            rows.append({
                "accn_rate": rate,
                "psnr_mean": r.get("mean_psnr"),
                "psnr_std":  r.get("std_psnr"),
                "ssim_mean": r.get("mean_ssim"),
                "ssim_std":  r.get("std_ssim"),
                "rmse_mean": r.get("mean_rmse"),
                "rmse_std":  r.get("std_rmse"),
                "count":     r.get("count"),
            })

        # Save CSV + JSON (no wandb)
        csv_path = os.path.join(out_dir, f"{exp_name}_test_{ts}.csv")
        json_path = os.path.join(out_dir, f"{exp_name}_test_{ts}.json")

        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
        except Exception:
            # Fallback if pandas isn’t available
            import csv
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["accn_rate"])
                writer.writeheader()
                writer.writerows(rows)

        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"\nSaved results:")
        print(f"  CSV : {csv_path}")
        print(f"  JSON: {json_path}")

        # Print final summary
        print("\nFinal test results:")
        for rate in sorted(all_results.keys()):
            r = all_results[rate]
            psnr = self._format_pm(r.get("mean_psnr"), r.get("std_psnr"), "{:.3f}", "{:.3f}")
            ssim = self._format_pm(r.get("mean_ssim"), r.get("std_ssim"), "{:.4f}", "{:.4f}")
            rmse = self._format_pm(r.get("mean_rmse"), r.get("std_rmse"), "{:.4f}", "{:.4f}")
            n = r.get("count")
            print(f"  accn_rate={rate}: PSNR {psnr} | SSIM {ssim} | RMSE {rmse} | N={n}")

        # Also print a compact table if pandas exists
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            # nicer formatting
            with pd.option_context(
                "display.max_rows", 200,
                "display.max_columns", 200,
                "display.width", 200,
            ):
                print("\nTable:")
                print(df.to_string(index=False))
        except Exception:
            pass

    # ---------- main ----------
    def __call__(self, model, data_module):
        # edit these as needed
        accelerations = [10, 20, 40, 80]
        if isinstance(accelerations, (int, float)):
            accelerations = [int(accelerations)]

        all_results: Dict[int, Dict[str, Any]] = {}

        for accn_rate in accelerations:
            accn_rate = int(accn_rate)

            if hasattr(model, "test_results"):
                delattr(model, "test_results")

            trainer = pl.Trainer(accelerator="gpu", devices=1, logger=False)
            _ = trainer.test(model, datamodule=data_module)

            if not hasattr(model, "test_results"):
                raise RuntimeError("model.test_results was not set. Check your test_step/on_test_epoch_end.")

            res = self._to_plain_dict(model.test_results)

            # Build per-rate dict (REMOVED zero-psnr fields)
            all_results[accn_rate] = {
                "mean_psnr": self._pick(res.get("mean_psnr")),
                "std_psnr":  self._pick(res.get("std_psnr")),
                "mean_ssim": self._pick(res.get("mean_ssim")),
                "std_ssim":  self._pick(res.get("std_ssim")),
                "mean_rmse": self._pick(res.get("mean_rmse")),
                "std_rmse":  self._pick(res.get("std_rmse")),
                "count":     self._pick(res.get("count")),
            }

            r = all_results[accn_rate]
            print(
                f"[TEST] accn_rate = {accn_rate} | "
                f"PSNR: {r['mean_psnr']:.3f} ± {r['std_psnr']:.3f}, "
                f"SSIM: {r['mean_ssim']:.4f} ± {r['std_ssim']:.4f}, "
                f"RMSE: {r['mean_rmse']:.4f} ± {r['std_rmse']:.4f} "
                f"(N={r['count']})"
            )

        # Save + print at end
        self.save_results(all_results)
        return all_results
