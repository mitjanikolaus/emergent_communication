import random

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pandas as pd
import numpy as np
import pytorch_lightning as pl


class SignalingGameDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, num_distractors: int, batch_size: int, num_workers: int):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = SignalingGameDiscriminationDataset(self.data_path, num_distractors)
        self.lang_analysis_dataset = SignalingGameDataset(self.data_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        language_analysis_dataloader = DataLoader(self.lang_analysis_dataset, batch_size=self.batch_size,
                                                  num_workers=self.num_workers)
        return language_analysis_dataloader


class SignalingGameDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, index_col=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].to_numpy()
        return sample.astype(np.float32)


class SignalingGameDiscriminationDataset(IterableDataset):
    def __init__(self, data_path, num_distractors):
        self.dataset = SignalingGameDataset(data_path)
        self.num_distractors = num_distractors

    def get_sample(self):
        target_position = random.choice(range(self.num_distractors))

        receiver_input = []
        for d in range(self.num_distractors):
            distractor = self.dataset[random.choice(range(len(self.dataset)))]
            receiver_input.append(distractor)

        receiver_input = torch.tensor(np.array(receiver_input))

        label = target_position
        sender_input = receiver_input[label]

        return sender_input, receiver_input, label

    def __iter__(self):
        while 1:
            yield self.get_sample()
