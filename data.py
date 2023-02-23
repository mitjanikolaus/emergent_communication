import itertools
import random

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class SignalingGameDataModule(pl.LightningDataModule):
    def __init__(self, num_attributes, num_values, max_num_objects, test_set_size, batch_size, num_workers, seed,
                 discrimination_game=False, num_objects=10, uninformative_attributes=False):
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
            self.train_dataset = SignalingGameDiscriminationDataset(objects_train, num_objects, max_num_objects, num_attributes, num_values, uninformative_attributes)
            self.val_dataset = SignalingGameDiscriminationDataset(objects_val, num_objects, max_num_objects, num_attributes, num_values, uninformative_attributes)
            self.test_dataset = SignalingGameDiscriminationDataset(objects_test, num_objects, max_num_objects, num_attributes, num_values, uninformative_attributes)
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
    if num_values**num_attributes <= max_num_objects:
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

    def __init__(self, objects, num_objects, max_samples, num_attributes, num_values, uninformative_attributes=False):
        self.num_objects = num_objects
        self.objects = objects
        self.max_samples = max_samples
        self.num_attributes = num_attributes
        self.num_values = num_values
        self.uninformative_attributes = uninformative_attributes

        self.candidate_combinations = []
        if self.uninformative_attributes:
            for i_1, j_1 in tqdm(list(itertools.product(range(self.num_attributes), range(self.num_values)))):
                filtered_objects = [o for o in self.objects if(o[i_1*self.num_values + j_1] == 1)]
                if len(filtered_objects) >= self.num_objects:
                    for i_2, j_2 in itertools.product(range(self.num_attributes), range(self.num_values)):
                        if not ((i_1 == i_2) and (j_1 == j_2)):
                            filtered_objects_2 = [o for o in filtered_objects if (o[i_2*self.num_values + j_2] == 1)]
                            if len(filtered_objects_2) >= self.num_objects:
                                self.candidate_combinations.append((i_1, j_1, i_2, j_2))
            print(f"num objects: {len(self.objects)} | num candidate combinations: {len(self.candidate_combinations)}")

    def get_sample(self):
        target_position = random.choice(range(self.num_objects))
        label = target_position
        if self.uninformative_attributes:
            (i_1, j_1, i_2, j_2) = random.choice(self.candidate_combinations)
            filtered_objects = [o for o in self.objects if
                                (o[i_1 * self.num_values + j_1] == 1) and (o[i_2 * self.num_values + j_2] == 1)]
            candidate_objects = random.sample(filtered_objects, self.num_objects)
        else:
            candidate_objects = random.sample(self.objects, self.num_objects)
        receiver_input = torch.stack(candidate_objects)
        sender_object = receiver_input[target_position]

        return sender_object, receiver_input, label

    def __iter__(self):
        for i in range(self.max_samples):
            yield self.get_sample()