import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pathlib
import re

class TrainVal:
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


    def __call__(self, model, data_module):

        checkpoint_callback = ModelCheckpoint(
            dirpath=self.cfg.exp_dir,
            filename='best',
            monitor=f'val_{model.val_test_loss.name}',
            mode=model.val_test_loss.mode,
            save_top_k=1,
            verbose=False,
        )

        # early_stop_callback = EarlyStopping(
        #     monitor=f'val_{model.val_test_loss.name}',
        #     patience=self.cfg.patience,
        #     mode=model.val_test_loss.mode,
        #     verbose=False
        # )

        additional_callbacks = []
        logger = WandbLogger(**dict(self.cfg.logger)) if self.cfg.get('logger') else None

        # ---- TRAIN/VAL on multiple GPUs ----

        train_trainer = pl.Trainer(
            accelerator='gpu',
            devices=self.cfg.trainer.num_devices,                     # your multi-GPU training
            logger=logger,
            callbacks=[
                checkpoint_callback,
                # early_stop_callback,
                *additional_callbacks
            ],
            max_epochs = self.cfg.trainer.max_epochs
        )
        
        shouldResume = getattr(self.cfg, 'resume', False)
        if shouldResume:
            ckpt = self._init_ckpt()
            train_trainer.fit(model, datamodule = data_module, ckpt_path = ckpt)
        else:
            train_trainer.fit(model, data_module)


        # ---- TEST on a single GPU ----
        # best_ckpt = checkpoint_callback.best_model_path  # explicit path
        # test_trainer = pl.Trainer(
        #     accelerator='gpu',
        #     devices=1,                    # force single-GPU test
        #     logger=logger,                # same W&B run
        #     callbacks=[],                 # no need for checkpoint callback here
        #     **{k: v for k, v in dict(self.cfg.trainer).items()
        #        if k not in ('devices', 'num_nodes', 'strategy')}
        # )
        # test_trainer.test(model, data_module, ckpt_path=best_ckpt)
