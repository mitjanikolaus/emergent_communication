import itertools
import random

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from tqdm import tqdm

RANDOM_STATE_TRAIN_TEST_SPLIT = 1

REQUEST = "REQUEST"
QUESTION_EXISTS = "QUESTION_EXISTS"
QUESTION_FORALL = "QUESTION_FORALL"


class SignalingGameDataModule(pl.LightningDataModule):
    def __init__(self, speech_acts, num_features, num_values, num_objects, max_num_objects, test_set_size, batch_size, num_workers, speech_acts_used=None):
        super().__init__()
        if speech_acts_used is None:
            speech_acts_used = speech_acts

        self.speech_acts_used = speech_acts_used
        self.num_features = num_features
        self.num_values = num_values
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_num_objects = max_num_objects

        objects = generate_objects(num_features, num_values, max_num_objects)
        objects_train, objects_test = train_test_split(objects, test_size=test_set_size, shuffle=True, random_state=RANDOM_STATE_TRAIN_TEST_SPLIT)
        print(f"Num objects in train: ", len(objects_train))
        print(f"Num objects in test: ", len(objects_test))

        self.train_data = dict()
        self.test_data = dict()
        for speech_act in self.speech_acts_used:
            if speech_act == REQUEST:
                data_train = objects_train
                data_test = objects_test
            elif speech_act == QUESTION_FORALL or speech_act == QUESTION_EXISTS:
                data_train = generate_question_contents(num_features, num_values)
                data_test = data_train
            else:
                raise ValueError(f"Unknown speech act: {speech_act}")

            self.train_data[speech_act] = data_train
            self.test_data[speech_act] = data_test

        self.train_dataset_discrimination = SignalingGameSpeechActsDiscriminationDataset(speech_acts_used, self.train_data, objects_train, num_objects, speech_acts)

        if len(objects_test) < 4:
            print("Small test data! Not possible to test generalization for question speech acts.")
            speech_acts_test = [sa for sa in speech_acts_used if sa not in [QUESTION_EXISTS, QUESTION_FORALL]]
            self.test_dataset_discrimination = SignalingGameSpeechActsDiscriminationDataset(speech_acts_test, self.test_data, objects_test, num_objects, speech_acts)
        else:
            self.test_dataset_discrimination = SignalingGameSpeechActsDiscriminationDataset(speech_acts_used, self.test_data, objects_test, num_objects, speech_acts)

        self.lang_analysis_dataset = SignalingGameLangAnalysisDataset(speech_acts_used, self.train_data, speech_acts)

    def train_dataloader(self):
        return DataLoader(self.train_dataset_discrimination, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        generalization_dataloader = DataLoader(self.test_dataset_discrimination, batch_size=self.batch_size,
                                               num_workers=self.num_workers)
        language_analysis_dataloader = DataLoader(self.lang_analysis_dataset, batch_size=self.batch_size,
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


def generate_question_contents(num_features, num_values):
    questions = []

    for feat in range(num_features):
        for val in range(num_values):
            z = torch.zeros((num_features, num_values))
            z[feat, val] = 1
            questions.append(z.view(-1))

    return questions


def generate_object(num_features, num_values):
    z = torch.zeros((num_features, num_values))
    for i in range(num_features):
        z[i, random.choice(range(num_values))] = 1
    return z.view(-1)


def generate_question_content(num_features, num_values):
    z = torch.zeros((num_features, num_values))
    z[random.choice(range(num_features)), random.choice(range(num_values))] = 1
    return z.view(-1)


class SignalingGameLangAnalysisDataset(Dataset):
    def __init__(self, speech_acts, datasets, all_speech_acts):
        self.speech_acts = speech_acts
        self.all_speech_acts = all_speech_acts
        self.datasets = datasets

        self.data = []
        for speech_act, contents in self.datasets.items():
            for content in contents:
                sender_input = torch.cat((speech_act_to_one_hot(speech_act, self.all_speech_acts), content))
                self.data.append(sender_input)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sender_input = self.data[idx]
        return sender_input


def get_speech_act(intent, speech_acts):
    speech_act_code = intent[:len(speech_acts)]
    return speech_acts[speech_act_code.nonzero().item()]


def get_speech_act_code(intent, num_speech_acts):
    return intent[:num_speech_acts]


def get_speech_act_codes(intents, num_speech_acts):
    return intents[:, :num_speech_acts]


def get_object(intent, speech_acts):
    return intent[len(speech_acts):]


def get_objects(intents, num_speech_acts):
    return intents[:, num_speech_acts:]


def speech_act_to_one_hot(speech_act, speech_acts):
    z = torch.zeros(len(speech_acts))
    z[speech_acts.index(speech_act)] = 1
    return z


def generate_requests(objects, speech_acts):
    requests = []
    sa_code = speech_act_to_one_hot(REQUEST, speech_acts)

    for object in objects:
        intent = torch.cat((sa_code, object))
        requests.append(intent)

    return requests


class SignalingGameSpeechActsDiscriminationDataset(IterableDataset):

    def __init__(self, speech_acts, datasets, objects, num_objects, all_speech_acts):
        self.speech_acts = speech_acts
        self.all_speech_acts = all_speech_acts
        self.label_true = num_objects
        self.label_false = num_objects + 1
        self.num_objects = num_objects
        self.datasets = datasets
        self.object_dataset = objects

        self.objects_matched = {}
        self.objects_not_matched = {}
        print("Generating matches:")
        for speech_act in self.speech_acts:
            if speech_act in [QUESTION_EXISTS, QUESTION_FORALL]:
                for question_content in tqdm(self.datasets[speech_act]):
                    matched = [d for d in self.object_dataset if torch.sum(d * question_content).item() > 0]
                    not_matched = [d for d in self.object_dataset if not torch.sum(d * question_content).item() > 0]
                    self.objects_matched[question_content] = matched
                    self.objects_not_matched[question_content] = not_matched

    def get_sample(self):
        speech_act = random.choice(self.speech_acts)

        if speech_act == REQUEST:
            target_position = random.choice(range(self.num_objects))
            label = target_position
            objects = random.sample(self.object_dataset, self.num_objects)
            receiver_input = torch.stack(objects)
            sender_object = receiver_input[target_position]
            sender_input = torch.cat((speech_act_to_one_hot(speech_act, self.all_speech_acts), sender_object))
            receiver_input[target_position] = get_object(sender_input, self.all_speech_acts)

        elif speech_act == QUESTION_EXISTS:
            question_content = random.choice(self.datasets[speech_act])
            sender_input = torch.cat((speech_act_to_one_hot(speech_act, self.all_speech_acts), question_content))

            # Ensure uniform distribution of labels
            label = random.choice([self.label_false, self.label_true])
            receiver_input = self.generate_objects_and_label_for_exists(self.object_dataset, question_content, label)

        elif speech_act == QUESTION_FORALL:
            question_content = random.choice(self.datasets[speech_act])
            sender_input = torch.cat((speech_act_to_one_hot(speech_act, self.all_speech_acts), question_content))

            # Ensure uniform distribution of labels
            label = random.choice([self.label_false, self.label_true])
            receiver_input = self.generate_objects_and_label_for_forall(self.object_dataset, question_content, label)

        else:
            raise ValueError("Unknown speech act: ", speech_act)

        return sender_input, receiver_input, label

    def __iter__(self):
        while 1:
            yield self.get_sample()

    def generate_objects_and_label_for_exists(self, data, question_content, label):
        if label == self.label_true:
            data_matched = self.objects_matched[question_content]
            objects = random.sample(data_matched, 1)
            objects.extend(random.sample(data, self.num_objects - 1))
        else:
            data_not_matched = self.objects_not_matched[question_content]
            objects = random.sample(data_not_matched, self.num_objects)

        receiver_input = torch.stack(objects)

        return receiver_input

    def generate_objects_and_label_for_forall(self, data, question_content, label):
        if label == self.label_true:
            data_matched = self.objects_matched[question_content]
            objects = random.sample(data_matched, self.num_objects)
        else:
            data_not_matched = self.objects_not_matched[question_content]
            objects = random.sample(data_not_matched, 1)
            objects.extend(random.sample(data, self.num_objects-1))

        receiver_input = torch.stack(objects)

        return receiver_input