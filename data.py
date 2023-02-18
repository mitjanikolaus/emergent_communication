import itertools
import random

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split


class SignalingGameDataModule(pl.LightningDataModule):
    def __init__(self, num_attributes, num_values, max_num_objects, test_set_size, batch_size, num_workers, seed,
                 discrimination_game=False, num_objects=10):
        super().__init__()
        self.num_attributes = num_attributes
        self.num_values = num_values
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_num_objects = max_num_objects
        self.num_objects = num_objects
        self.discrimination_game = discrimination_game

        objects = generate_objects(num_attributes, num_values, max_num_objects)
        if test_set_size > 0:
            objects_train, objects_test = train_test_split(objects, test_size=test_set_size, shuffle=True, random_state=seed)
            objects_train, objects_val = train_test_split(objects_train, test_size=test_set_size, shuffle=True, random_state=seed)
        else:
            print("Test set size <= 0: Setting train_set = test_set = val_set!")
            objects_train = objects
            objects_val = objects
            objects_test = objects
        print(f"Num objects in train: ", len(objects_train))
        print(f"Num objects in val: ", len(objects_val))
        print(f"Num objects in test: ", len(objects_test))

        if self.discrimination_game:
            self.train_dataset = SignalingGameDiscriminationDataset(objects_train, num_objects, max_num_objects)
            self.val_dataset = SignalingGameDiscriminationDataset(objects_val, num_objects, max_num_objects)
            self.test_dataset = SignalingGameDiscriminationDataset(objects_test, num_objects, max_num_objects)
        else:
            self.train_dataset = SignalingGameDataset(objects_train)
            self.val_dataset = SignalingGameDataset(objects_val)
            self.test_dataset = SignalingGameDataset(objects_test)

    def train_dataloader(self):
        shuffle = True
        if self.discrimination_game:
            shuffle = False
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def val_dataloader(self):
        validation_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                               num_workers=self.num_workers)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        language_analysis_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                  num_workers=self.num_workers)
        return validation_dataloader, test_dataloader, language_analysis_dataloader


def generate_objects(num_attributes, num_values, max_num_objects):
    samples = set()
    if num_values**num_attributes < max_num_objects:
        inputs = itertools.product(range(num_values), repeat=num_attributes)
        for input in inputs:
            z = torch.zeros((num_attributes, num_values))
            for i in range(num_attributes):
                z[i, input[i]] = 1
            samples.add(z.view(-1))
    else:
        while len(samples) < max_num_objects:
            z = torch.zeros((num_attributes, num_values))
            for i in range(num_attributes):
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


class SignalingGameDiscriminationDataset(IterableDataset):

    def __init__(self, objects, num_objects, max_samples):
        self.num_objects = num_objects
        self.objects = objects
        self.max_samples = max_samples

    def get_sample(self):
        target_position = random.choice(range(self.num_objects))
        label = target_position
        candidate_objects = random.sample(self.objects, self.num_objects)
        receiver_input = torch.stack(candidate_objects)
        sender_object = receiver_input[target_position]

        return sender_object, receiver_input, label

    def __iter__(self):
        for i in range(self.max_samples):
            yield self.get_sample()