import argparse
import os
from src.config import Configurator
import wandb
from pytorch_lightning import seed_everything

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-resolution CT')
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to config file"
    )
    parser.add_argument(
        "--fix", "-f",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options using the command-line (e.g. python3 demo.py --config-file ***.py --opts key1 val1 key2 val2)"
    )
    args = parser.parse_args()

    seed_everything(42, workers=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    # local_rank = int(os.environ.get("SLURM_PROCID"))
    print(f"local_rank: {local_rank}", flush=True)

    if args.fix is None:
        args.fix = []

    # Initialize config from YAML + CLI overrides
    cc = Configurator(args)

    # Create experiment directory and save final config
    os.makedirs(cc.cfg.exp_dir, exist_ok=True)
    with open(f'{cc.cfg.exp_dir}/config.yaml', 'w') as f:
        f.write(str(cc.cfg))

    # Initialize W&B
    wandb_mode = "online" if local_rank == 0 else "disabled"
    # wandb_mode = "disabled"
    wandb.init(mode=wandb_mode)

    # Set W&B run name from config (only on rank 0)
    if local_rank == 0 and hasattr(cc.cfg, "exp_name"):
        wandb.run.name = cc.cfg.exp_name

    # Build experiment and run
    exp, model, data_module = cc.init_all()
    exp(model, data_module)
