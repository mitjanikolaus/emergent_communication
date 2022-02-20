import itertools
import random
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import pytorch_lightning as pl


class SignalingGameDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        dataset = SignalingGameDiscriminationDataset(self.data_path)
        self.train_dataset, self.val_dataset = random_split(dataset, [round(len(dataset)*0.9), round(len(dataset)*0.1)])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)


class SignalingGameDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, index_col=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].to_numpy()
        return sample.astype(np.float32)


class SignalingGameDiscriminationDataset(Dataset):
    def __init__(self, data_path):
        self.dataset = SignalingGameDataset(data_path)

        self.data = list(itertools.permutations(self.dataset, 2))

    def __len__(self):
        return len(self.data) * 2

    def __getitem__(self, idx):
        data_idx = idx % len(self.data)
        receiver_input = self.data[data_idx]
        receiver_input = torch.tensor(np.array(receiver_input))

        target_position = idx % 2
        label = target_position
        sender_input = receiver_input[label]

        return sender_input, receiver_input, label
