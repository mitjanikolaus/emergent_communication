import itertools
import random

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

RANDOM_STATE_TRAIN_TEST_SPLIT = 1


class SignalingGameDataModule(pl.LightningDataModule):
    def __init__(self, num_features, num_values, max_num_objects, test_set_size, batch_size, num_workers):
        super().__init__()
        self.num_features = num_features
        self.num_values = num_values
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_num_objects = max_num_objects

        objects = generate_objects(num_features, num_values, max_num_objects)
        objects_train, objects_test = train_test_split(objects, test_size=test_set_size, shuffle=True, random_state=RANDOM_STATE_TRAIN_TEST_SPLIT)
        print(f"Num objects in train: ", len(objects_train))
        print(f"Num objects in test: ", len(objects_test))

        self.train_dataset = SignalingGameDataset(objects_train)
        self.test_dataset = SignalingGameDataset(objects_test)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        generalization_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                               num_workers=self.num_workers)
        language_analysis_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                  num_workers=self.num_workers)
        return generalization_dataloader, language_analysis_dataloader


def generate_objects(num_features, num_values, max_num_objects):
    samples = set()
    if num_values**num_features < max_num_objects:
        inputs = itertools.product(range(num_values), repeat=num_features)
        for input in inputs:
            z = torch.zeros((num_features, num_values))
            for i in range(num_features):
                z[i, input[i]] = 1
            samples.add(z.view(-1))
    else:
        while len(samples) < max_num_objects:
            z = torch.zeros((num_features, num_values))
            for i in range(num_features):
                value = random.choice(range(num_values))
                z[i, value] = 1
            samples.add(z.view(-1))

    return list(samples)


class SignalingGameDataset(Dataset):

    def __init__(self, objects):
        self.objects = objects

    def __getitem__(self, id):
        object = self.objects[id]

        return object

    def __len__(self):
        return len(self.objects)
