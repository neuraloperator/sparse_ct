"""Download pretrained CTO model weights from the Hugging Face Hub.

The repo (aujasvit/cto) has two folders:
    aapm/cto_aapm.ckpt
    kits/cto_kits.ckpt

Usage:
    python download_weights.py                       # -> ./weights
    python download_weights.py --local-dir weights   # explicit target
    python download_weights.py --token hf_xxx         # private repo

For a private repo you can instead run `hf auth login` (or set the HF_TOKEN
env var) once, and omit --token.

Then run the test script against the downloaded checkpoints, e.g.:
    python scripts/test.py -c configs/aapm.yaml --fix init_exp_dir weights/aapm --no-wandb
    python scripts/test.py -c configs/kits.yaml --fix init_exp_dir weights/kits --no-wandb
"""
import argparse
from huggingface_hub import snapshot_download

DEFAULT_REPO = "aujasvit/cto"


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained CTO weights from the Hugging Face Hub."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO,
                        help="HF model repo id (default: %(default)s)")
    parser.add_argument("--local-dir", default="weights",
                        help="Local directory to download into (default: %(default)s)")
    parser.add_argument("--token", default=None,
                        help="HF token for private repos (or use `hf auth login` / HF_TOKEN env var)")
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="model",
        local_dir=args.local_dir,
        token=args.token,
    )

    print(f"\nDownloaded weights to: {path}")
    print("Run test against them, e.g.:")
    print(f"  python scripts/test.py -c configs/aapm.yaml --fix init_exp_dir {args.local_dir}/aapm --no-wandb")
    print(f"  python scripts/test.py -c configs/kits.yaml --fix init_exp_dir {args.local_dir}/kits --no-wandb")


if __name__ == "__main__":
    main()
