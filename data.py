import itertools
import random

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl


class SignalingGameDataModule(pl.LightningDataModule):
    def __init__(self, num_features, num_values, num_distractors, test_set_size, batch_size, num_workers):
        super().__init__()
        self.num_features = num_features
        self.num_values = num_values
        self.batch_size = batch_size
        self.num_workers = num_workers

        dataset = SignalingGameDataset(num_features, num_values)
        num_test_samples = round(len(dataset)*test_set_size)
        num_train_samples = len(dataset) - num_test_samples
        self.data_train, self.data_test = torch.utils.data.random_split(dataset, [num_train_samples, num_test_samples])
        print("Num meanings in train: ", len(self.data_train))
        print("Num meanings in test: ", len(self.data_test))

        self.train_dataset_discrimination = SignalingGameDiscriminationDataset(self.data_train, num_distractors)
        self.test_dataset_discrimination = SignalingGameDiscriminationDataset(self.data_test, num_distractors)

    def train_dataloader(self):
        return DataLoader(self.train_dataset_discrimination, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        generalization_dataloader = DataLoader(self.test_dataset_discrimination, batch_size=self.batch_size,
                                               num_workers=self.num_workers)
        language_analysis_dataloader = DataLoader(self.data_train, batch_size=self.batch_size,
                                                  num_workers=self.num_workers)
        return generalization_dataloader, language_analysis_dataloader

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if dataloader_idx == 0:
            sender_input, receiver_input, target_position = batch
            sender_input = sender_input.to(device)
            receiver_input = receiver_input.to(device)
            target_position = target_position.to(device)
            return sender_input, receiver_input, target_position
        else:
            sender_input = batch
            sender_input = sender_input.to(device)
            return sender_input


def generate_data(num_features, num_values):
    inputs = itertools.product(range(num_values), repeat=num_features)

    samples = []
    for input in inputs:
        z = torch.zeros((num_features, num_values))
        for i in range(num_features):
            z[i, input[i]] = 1
        samples.append(z.view(-1))

    return samples


class SignalingGameDataset(Dataset):
    def __init__(self, num_features, num_values):
        self.data = generate_data(num_features, num_values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample


class SignalingGameDiscriminationDataset(IterableDataset):
    def __init__(self, dataset, num_distractors):
        self.dataset = dataset
        self.num_distractors = num_distractors

    def get_sample(self):
        receiver_input = []
        for d in range(self.num_distractors):
            distractor = self.dataset[random.choice(range(len(self.dataset)))]
            receiver_input.append(distractor)

        receiver_input = torch.stack(receiver_input)

        target_position = random.choice(range(self.num_distractors))
        sender_input = receiver_input[target_position]

        return sender_input, receiver_input, target_position

    def __iter__(self):
        while 1:
            yield self.get_sample()
