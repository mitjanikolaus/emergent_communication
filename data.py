import itertools
import os
import random

import h5py
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from utils import H5_IDS_KEY, DATA_DIR_GUESSWHAT, DATA_DIR_IMAGENET


class SignalingGameDataModule(pl.LightningDataModule):
    def __init__(self, num_attributes, num_values, max_num_objects, test_set_size, batch_size, num_workers, seed,
                 discrimination_game=False, num_objects=10, hard_distractors=False, guesswhat=False, imagenet=False):
        super().__init__()
        self.num_attributes = num_attributes
        self.num_values = num_values
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_num_objects = max_num_objects
        self.num_objects = num_objects
        self.discrimination_game = discrimination_game
        self.imagenet = imagenet

        objects = generate_objects(num_attributes, num_values, max_num_objects)
        if test_set_size > 0:
            objects_train, objects_test = train_test_split(objects, test_size=test_set_size, shuffle=True, random_state=seed)
            objects_train, objects_val = train_test_split(objects_train, test_size=test_set_size, shuffle=True, random_state=seed)
        else:
            print("Test set size <= 0: Setting train_set = test_set = val_set!")
            objects_train = objects
            objects_val = objects
            objects_test = objects

        if guesswhat:
            print("GuessWhat Game")
            self.train_dataset = SignalingGameGuessWhatDataset("train_features.hdf5", num_objects)
            val_dataset_file = "validation_features.hdf5"
            if test_set_size <= 0:
                val_dataset_file = "train_features.hdf5"
            self.val_dataset = SignalingGameGuessWhatDataset(val_dataset_file, num_objects)
            self.test_dataset = None

        elif imagenet:
            print("ImageNet Game")
            self.train_dataset = SignalingGameImagenetDataset("train_features.hdf5", num_objects)
            val_dataset_file = "val_features.hdf5"
            if test_set_size <= 0:
                val_dataset_file = "train_features.hdf5"
            self.val_dataset = SignalingGameImagenetDataset(val_dataset_file, num_objects)
            self.test_dataset = None

        else:
            print(f"Num objects in train: ", len(objects_train))
            print(f"Num objects in val: ", len(objects_val))
            print(f"Num objects in test: ", len(objects_test))

            if self.discrimination_game:
                self.train_dataset = SignalingGameDiscriminationDataset(objects_train, num_objects, max_num_objects, num_attributes, num_values, hard_distractors)
                self.val_dataset = SignalingGameDiscriminationDataset(objects_val, num_objects, max_num_objects, num_attributes, num_values, hard_distractors)
                self.test_dataset = SignalingGameDiscriminationDataset(objects_test, num_objects, max_num_objects, num_attributes, num_values, hard_distractors)
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
        collate_fn = None
        if self.imagenet:
            collate_fn = self.collate_add_batch_distractors

        validation_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                               num_workers=self.num_workers, collate_fn=collate_fn)
        language_analysis_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                  num_workers=self.num_workers, collate_fn=collate_fn)
        if self.test_dataset:
            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                         collate_fn=collate_fn)

            return validation_dataloader, language_analysis_dataloader, test_dataloader
        else:
            return validation_dataloader, language_analysis_dataloader

    def collate_add_batch_distractors(self, batch):
        sender_objects = []
        receiver_inputs = []
        labels = []
        for i, target_object in enumerate(batch):
            candidate_object_ids = random.sample(range(len(batch)), k=self.num_objects)
            while i in candidate_object_ids:
                # Ensure that target is not among candidates
                candidate_object_ids = random.sample(range(len(batch)), k=self.num_objects)

            target_position = random.choice(range(len(candidate_object_ids)))

            candidate_objects = [batch[idx] for idx in candidate_object_ids]
            candidate_object_ids[target_position] = target_object

            receiver_input = torch.stack(candidate_objects)
            sender_object = receiver_input[target_position]

            sender_objects.append(sender_object)
            receiver_inputs.append(receiver_inputs)
            labels.append(target_position)

        receiver_inputs = torch.stack(receiver_inputs)
        sender_objects = torch.stack(sender_objects)
        labels = torch.stack(labels)

        return sender_objects, receiver_inputs, labels



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


class SignalingGameGuessWhatDataset(Dataset):

    def __init__(self, file_name, num_objects):
        self.file_name = file_name

        self.num_objects = num_objects

        self.h5_db = h5py.File(os.path.join(DATA_DIR_GUESSWHAT, self.file_name), 'r')
        self.h5_ids = self.h5_db[H5_IDS_KEY]

    def __len__(self):
        return len(self.h5_ids)

    def __getitem__(self, index):
        candidate_objects = self.h5_db[self.h5_ids[index]]

        # Discard first image (scene overview)
        candidate_objects = candidate_objects[1:]

        random.shuffle(candidate_objects)
        candidate_objects = candidate_objects[:self.num_objects]

        target_position = random.choice(range(len(candidate_objects)))
        label = target_position

        candidate_objects = [torch.tensor(o) for o in candidate_objects]

        # Pad with 0 objects
        candidate_objects += [torch.zeros_like(candidate_objects[0])] * (self.num_objects - len(candidate_objects))

        receiver_input = torch.stack(candidate_objects)
        sender_object = receiver_input[target_position]

        return sender_object, receiver_input, label


class SignalingGameImagenetDataset(Dataset):

    def __init__(self, file_name, num_objects):
        self.file_name = file_name

        self.num_objects = num_objects

        self.h5_db = h5py.File(os.path.join(DATA_DIR_IMAGENET, self.file_name), 'r')
        self.h5_ids = list(self.h5_db[H5_IDS_KEY])

    def __len__(self):
        return len(self.h5_ids)

    def __getitem__(self, index):
        target_id = self.h5_ids[index]

        return torch.tensor(self.h5_db[target_id])


class SignalingGameDiscriminationDataset(IterableDataset):

    def __init__(self, objects, num_objects, max_samples, num_attributes, num_values, hard_distractors=False):
        self.num_objects = num_objects
        self.objects = objects
        self.max_samples = max_samples
        self.num_attributes = num_attributes
        self.num_values = num_values
        self.hard_distractors = hard_distractors

    def get_sample(self):
        target_position = random.choice(range(self.num_objects))
        label = target_position
        if self.hard_distractors:
            target_object = random.choice(self.objects)
            candidate_objects = []
            while len(candidate_objects) < self.num_objects:
                attrs_uninformative = random.sample(range(self.num_attributes), k=self.num_attributes - 1)
                distractor = torch.zeros((self.num_attributes, self.num_values))
                for attr in range(self.num_attributes):
                    if attr in attrs_uninformative:
                        # Take the target objects value
                        val = torch.nonzero(target_object[attr * self.num_values:(attr + 1) * self.num_values])[0]
                    else:
                        val = random.choice(range(self.num_values))
                        # Ensure that the value is different from the target objects value
                        while val == torch.nonzero(target_object[attr * self.num_values:(attr + 1) * self.num_values])[0].item():
                            val = random.choice(range(self.num_values))

                    distractor[attr, val] = 1

                distractor = distractor.view(-1)
                candidate_objects.append(distractor)

            candidate_objects[target_position] = target_object
        else:
            candidate_objects = random.sample(self.objects, self.num_objects)
        receiver_input = torch.stack(candidate_objects)
        sender_object = receiver_input[target_position]

        return sender_object, receiver_input, label

    def __iter__(self):
        for i in range(self.max_samples):
            yield self.get_sample()