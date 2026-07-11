import argparse
import random
import os
from src.config import TestConfigurator
import wandb
import pandas as pd
from pytorch_lightning import seed_everything

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-resolution CT')
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config file"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Skip W&B logging entirely. Metrics are still saved to results/ (CSV + JSON)."
    )
    parser.add_argument(
        "--fix", "-f",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line (e.g. python3 demo.py --config-file ***.py --opts key1 val1 key2 val2)"
    )
    args = parser.parse_args()

    seed_everything(42, workers=True)

    if args.fix is None:
        args.fix = []

    # Initialize config from YAML + CLI overrides
    cc = TestConfigurator(args)

    # Create experiment directory and save final config
    os.makedirs(cc.cfg.exp_dir, exist_ok=True)
    with open(f'{cc.cfg.exp_dir}/config.yaml', 'w') as f:
        f.write(str(cc.cfg))

    exp, model, data_module = cc.init_all()
    all_results = exp(model, data_module)

    # ---- Optionally log results to W&B as a table ----
    # The engine already wrote CSV + JSON to results/, so W&B is purely optional.
    # Skipped cleanly on --no-wandb, WANDB_MODE=disabled, or any wandb error
    # (e.g. not logged in) so a reviewer can run test with zero wandb setup.
    if args.no_wandb or not cc.cfg.get("use_wandb", True) or os.environ.get("WANDB_MODE", "").lower() == "disabled":
        print("[TEST] W&B logging skipped. Metrics saved to results/ (CSV + JSON).")
    else:
        try:
            cfg_logger = cc.cfg.get("logger")
            wb_kwargs = {}
            if cfg_logger is not None and isinstance(cfg_logger, dict) and len(cfg_logger) > 1:
                wb_kwargs = {k: v for k, v in dict(cfg_logger).items() if k != "save_dir"}
            # project = original project + _test  (e.g. "eccv" -> "eccv_test")
            original_project = wb_kwargs.pop("project", None) or wb_kwargs.pop("name", cc.cfg.exp_name)
            wb_kwargs["project"] = f"{original_project}_test"
            wb_kwargs["name"] = f"{cc.cfg.exp_name}_test"
            wb_kwargs["job_type"] = "test"
            # Honor WANDB_MODE (online/offline), default online. Resolve entity:
            # WANDB_ENTITY env wins, then config; drop the "your_entity" placeholder
            # so wandb uses the logged-in default account.
            wb_kwargs["mode"] = os.environ.get("WANDB_MODE", "online")
            entity = os.environ.get("WANDB_ENTITY") or wb_kwargs.get("entity")
            if entity in (None, "", "your_entity"):
                wb_kwargs.pop("entity", None)
            else:
                wb_kwargs["entity"] = entity

            run = wandb.init(**wb_kwargs)

            # Build a DataFrame from results
            rows = []
            for rate in sorted(all_results.keys()):
                r = all_results[rate]
                rows.append({
                    "accn_rate": int(rate),
                    "psnr_mean": float(r.get("mean_psnr", 0)),
                    "psnr_std":  float(r.get("std_psnr", 0)),
                    "ssim_mean": float(r.get("mean_ssim", 0)),
                    "ssim_std":  float(r.get("std_ssim", 0)),
                    "rmse_mean": float(r.get("mean_rmse", 0)),
                    "rmse_std":  float(r.get("std_rmse", 0)),
                    "count":     int(r.get("count", 0)),
                })
            df = pd.DataFrame(rows)
            # Re-seed from OS entropy before logging the table. seed_everything(42)
            # above pins Python's random, which wandb uses to mint the table artifact's
            # id; without this the id collides and the table never renders in the
            # dashboard. Do NOT remove — this is a wandb-visibility fix, not sampling.
            random.seed(None)
            # Log table from DataFrame
            table = wandb.Table(dataframe=df)
            run.log({"test_results": table})

            # Also log each rate's metrics as scalars (always visible in dashboard)
            for _, row in df.iterrows():
                rate = int(row["accn_rate"])
                run.summary[f"rate_{rate}/psnr_mean"] = row["psnr_mean"]
                run.summary[f"rate_{rate}/psnr_std"]  = row["psnr_std"]
                run.summary[f"rate_{rate}/ssim_mean"] = row["ssim_mean"]
                run.summary[f"rate_{rate}/ssim_std"]  = row["ssim_std"]
                run.summary[f"rate_{rate}/rmse_mean"] = row["rmse_mean"]
                run.summary[f"rate_{rate}/rmse_std"]  = row["rmse_std"]
                run.summary[f"rate_{rate}/count"]     = row["count"]

            run.finish()
            print("W&B run finished — table is visible at:", run.url)
        except Exception as e:
            print(f"[TEST] W&B logging skipped ({type(e).__name__}: {e}). "
                  f"Metrics saved to results/ (CSV + JSON).")
