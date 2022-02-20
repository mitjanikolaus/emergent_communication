import itertools
import random

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class SignalingGameDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, index_col=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx].to_numpy()
        return sample.astype(np.float32)


class SignalingGameDiscriminationDataset(Dataset):
    def __init__(self, data_path, num_distractors=1):
        self.dataset = SignalingGameDataset(data_path)

        self.num_distractors = num_distractors
        self.data = list(itertools.combinations(self.dataset, num_distractors+1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        receiver_input = self.data[idx]
        random.shuffle(list(receiver_input))
        receiver_input = torch.tensor(np.array(receiver_input))

        correct_position = random.choice(range(self.num_distractors+1))

        label = correct_position

        sender_input = receiver_input[label]

        return sender_input, receiver_input, label
