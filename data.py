import itertools
import os
import pickle
import random

import h5py
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from utils import DATA_DIR_GUESSWHAT, DATA_DIR_IMAGENET


class SignalingGameDataModule(pl.LightningDataModule):
    def __init__(self, num_attributes, num_values, max_num_objects, val_set_size, test_set_size, batch_size,
                 num_workers, seed, num_objects=10, hard_distractors=False, guesswhat=False,
                 imagenet=False):
        super().__init__()
        self.num_attributes = num_attributes
        self.num_values = num_values
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_num_objects = max_num_objects
        self.num_objects = num_objects
        self.imagenet = imagenet
        self.use_test_set = test_set_size > 0
        self.hard_distractors = hard_distractors

        self.collate_fn = None
        if self.imagenet and not self.hard_distractors:
            self.collate_fn = self.collate_add_batch_distractors

        objects = generate_objects(num_attributes, num_values, max_num_objects)
        if self.use_test_set:
            objects_train, objects_test = train_test_split(objects, test_size=test_set_size, shuffle=True, random_state=seed)
            objects_train, objects_val = train_test_split(objects_train, test_size=val_set_size, shuffle=True, random_state=seed)
        elif val_set_size > 0:
            objects_train, objects_val = train_test_split(objects, test_size=val_set_size, shuffle=True, random_state=seed)
            objects_test = None
        else:
            print("Val set size <= 0: Setting train_set = val_set!")
            objects_train = objects
            objects_val = objects
            objects_test = None

        if guesswhat:
            print("GuessWhat Game")
            self.train_dataset = SignalingGameGuessWhatDataset("train_features.hdf5", num_objects)
            val_dataset_file = "validation_features.hdf5"
            if val_set_size <= 0:
                val_dataset_file = "train_features.hdf5"
            self.val_dataset = SignalingGameGuessWhatDataset(val_dataset_file, num_objects)
            self.test_dataset = None

        elif imagenet:
            print("ImageNet Game")
            self.train_dataset = SignalingGameImagenetDataset("val_features.hdf5", num_objects, hard_distractors) # TODO testing with val
            val_dataset_file = "val_features.hdf5"
            if val_set_size <= 0:
                val_dataset_file = "train_features.hdf5"
            self.val_dataset = SignalingGameImagenetDataset(val_dataset_file, num_objects, hard_distractors)
            self.test_dataset = None

        else:
            print(f"Num objects in train: ", len(objects_train))
            print(f"Num objects in val: ", len(objects_val))
            if self.use_test_set:
                print(f"Num objects in test: ", len(objects_test))

            self.train_dataset = SignalingGameDiscriminationDataset(objects_train, num_objects, max_num_objects, num_attributes, num_values, hard_distractors)
            self.val_dataset = SignalingGameDiscriminationDataset(objects_val, num_objects, max_num_objects, num_attributes, num_values, hard_distractors)
            if self.use_test_set:
                self.test_dataset = SignalingGameDiscriminationDataset(objects_test, num_objects, max_num_objects, num_attributes, num_values, hard_distractors)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=False, collate_fn=self.collate_fn)

    def val_dataloader(self):
        validation_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size,
                                               num_workers=self.num_workers, collate_fn=self.collate_fn)
        language_analysis_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                  num_workers=self.num_workers, collate_fn=self.collate_fn)
        if self.use_test_set:
            test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                                         collate_fn=self.collate_fn)

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
            receiver_inputs.append(receiver_input)
            labels.append(torch.tensor(target_position))

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
        self.h5_ids = list(self.h5_db.keys())

    def __len__(self):
        return len(self.h5_ids)

    def __getitem__(self, index):
        candidate_objects = self.h5_db[self.h5_ids[index]]

        # Discard first image (scene overview)
        candidate_objects = candidate_objects[1:]

        candidate_objects = candidate_objects[torch.randperm(len(candidate_objects))]
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

    def __init__(self, file_name, num_objects, hard_distractors=False):
        self.file_name = file_name

        self.num_objects = num_objects

        self.hard_distractors = hard_distractors
        if hard_distractors:
            self.nns = pickle.load(open(os.path.join(DATA_DIR_IMAGENET, "nearest_neighors.p"), "rb"))

        self.h5_db = h5py.File(os.path.join(DATA_DIR_IMAGENET, self.file_name), 'r')
        self.h5_ids = self.h5_db[H5_IDS_KEY]

    def __len__(self):
        # TODO: fix for bug in extract feats:
        return len(self.h5_ids) - len(self.h5_ids) % 100

    def __getitem__(self, index):
        if self.hard_distractors:
            target_id = self.h5_ids[index]

            candidate_object_ids = self.nns[target_id]

            random.shuffle(candidate_object_ids)
            candidate_object_ids = candidate_object_ids[:self.num_objects]

            target_position = random.choice(range(len(candidate_object_ids)))
            label = target_position

            candidate_objects = [torch.tensor(self.h5_db[id]) for id in candidate_object_ids]

            receiver_input = torch.stack(candidate_objects)
            sender_object = receiver_input[target_position]

            return sender_object, receiver_input, label

        else:
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
            candidate_objects = [target_object]

            attr_informative = random.choice(range(self.num_attributes))

            # Values that are already taken
            distractor_values = set()
            # Add the target value
            target_attr_inf_value = torch.nonzero(target_object[attr_informative * self.num_values:(attr_informative + 1) * self.num_values])[0].item()
            distractor_values.add(target_attr_inf_value)
            while len(candidate_objects) < self.num_objects:
                distractor = torch.zeros((self.num_attributes, self.num_values))
                for attr in range(self.num_attributes):
                    if attr != attr_informative:
                        # Take the target objects value
                        val = torch.nonzero(target_object[attr * self.num_values:(attr + 1) * self.num_values])[0]
                    else:
                        # Ensure that the value is different from the target objects value
                        possible_values = set(range(self.num_values)) - distractor_values
                        if len(possible_values) == 0:
                            raise RuntimeError(f"Not enough values ({self.num_values}) available to create sufficient distractors ({self.num_objects})")
                        val = random.choice(list(possible_values))
                        distractor_values.add(val)

                    distractor[attr, val] = 1

                distractor = distractor.view(-1)
                candidate_objects.append(distractor)

            candidate_objects = list(candidate_objects)
        else:
            candidate_objects = random.sample(self.objects, self.num_objects)
        receiver_input = torch.stack(candidate_objects)
        sender_object = receiver_input[target_position]

        return sender_object, receiver_input, label

    def __iter__(self):
        for i in range(self.max_samples):
            yield self.get_sample()