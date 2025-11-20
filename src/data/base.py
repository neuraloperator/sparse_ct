from abc import ABC, abstractmethod
from pytorch_lightning import LightningDataModule
from dataclasses import dataclass
from torch.utils.data import DataLoader
import numpy as np


@dataclass
class BaseDataModule(LightningDataModule, ABC):
    def __init__(self, shape, batch_size):
        super().__init__()
        self.shape = shape
        self.batch_size = batch_size

    def train_dataloader(self):
        print(f'-------> training set: {int(np.ceil(len(self.train_dataset)/self.batch_size))} batches of size {self.batch_size} ({len(self.train_dataset)} samples in total) <-------')
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            num_workers=12, 
            shuffle=True, 
            drop_last=False,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            )

    def val_dataloader(self):
        print(f'-------> validation set: {len(self.val_dataset)} batches of size 1 ({len(self.val_dataset)} samples in total) <-------')
        return DataLoader(
            self.val_dataset, 
            batch_size=1, 
            num_workers=12, 
            shuffle=False, 
            drop_last=False,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            )
    
    def test_dataloader(self):
        print(f'-------> test set: {len(self.test_dataset)} batches of size 1 ({len(self.test_dataset)} samples in total) <-------')
        return DataLoader(
            self.test_dataset, 
            batch_size=1, 
            num_workers=12, 
            shuffle=False, 
            drop_last=False,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            )

    def teardown(self, stage=None):
        # Used to clean-up when the run is finished
        pass
