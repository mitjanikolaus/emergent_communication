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

        # self.train_datasets = []
        # self.test_datasets = []
        # for speech_act in SPEECH_ACTS:
        #     dataset = SignalingGameSpeechActDataset(num_features, num_values, speech_act)
        #     num_test_samples = round(len(dataset)*test_set_size)
        #     num_train_samples = len(dataset) - num_test_samples
        #     data_train, data_test = torch.utils.data.random_split(dataset, [num_train_samples, num_test_samples])
        #     print("Num meanings in train: ", len(data_train))
        #     print("Num meanings in test: ", len(data_test))
        #     self.train_datasets.append(data_train)
        #     self.test_datasets.append(data_test)

        # TODO:
        # self.language_analysis_dataset = SignalingGameSpeechActDataset(num_features, num_values, REQUEST)


        self.train_dataset_discrimination = SignalingGameSpeechActsDiscriminationDataset(self.num_features, self.num_values, num_distractors, SPEECH_ACTS)
        # TODO: currently train = test
        self.test_dataset_discrimination = SignalingGameSpeechActsDiscriminationDataset(self.num_features, self.num_values, num_distractors, SPEECH_ACTS)
        self.test_dataset_language_analysis = SignalingGameSpeechActsDiscriminationDataset(self.num_features, self.num_values, num_distractors, [REQUEST])


    def train_dataloader(self):
        return DataLoader(self.train_dataset_discrimination, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        generalization_dataloader = DataLoader(self.test_dataset_discrimination, batch_size=self.batch_size,
                                               num_workers=self.num_workers)
        # language_analysis_dataloader = DataLoader(self.language_analysis_dataset, batch_size=self.batch_size,
        #                                           num_workers=self.num_workers)
        language_analysis_dataloader = DataLoader(self.test_dataset_language_analysis, batch_size=self.batch_size,
                                               num_workers=self.num_workers)
        return generalization_dataloader, language_analysis_dataloader

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # if dataloader_idx == 0:
        sender_input, receiver_input, target_position = batch
        sender_input = sender_input.to(device)
        receiver_input = receiver_input.to(device)
        target_position = target_position.to(device)
        return sender_input, receiver_input, target_position
        # else:
        #     sender_input = batch
        #     sender_input = sender_input.to(device)
        #     return sender_input


def generate_objects(num_features, num_values):
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
        self.data = generate_objects(num_features, num_values)

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

REQUEST = "REQUEST"
QUESTION_EXISTS = "EXISTS"
QUESTION_FORALL = "FORALL"
# QUESTION_WH = "WHICH"
SPEECH_ACTS = [REQUEST, QUESTION_EXISTS, QUESTION_FORALL]


def speech_act_to_one_hot(speech_act):
    z = torch.zeros(len(SPEECH_ACTS))
    z[SPEECH_ACTS.index(speech_act)] = 1
    return z


def generate_requests(objects):
    requests = []
    sa_code = speech_act_to_one_hot(REQUEST)

    for object in objects:
        intent = torch.cat((sa_code, object))
        requests.append(intent)

    return requests


def generate_questions(num_features, num_values, speech_act):
    questions = []
    sa_code = speech_act_to_one_hot(speech_act)

    for feat in range(num_features):
        for val in range(num_values):
            z = torch.zeros((num_features, num_values))
            z[feat, val] = 1
            intent = torch.cat((sa_code, z.view(-1)))
            questions.append(intent)

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


class SignalingGameSpeechActDataset(Dataset):
    def __init__(self, num_features, num_values, speech_act_type):
        self.speech_act_type = speech_act_type
        if speech_act_type == REQUEST:
            self.objects = generate_objects(num_features, num_values)
            self.intents = generate_requests(self.objects)
        elif speech_act_type == QUESTION_FORALL or speech_act_type == QUESTION_EXISTS:
            self.intents = generate_questions(num_features, num_values, speech_act_type)
        else:
            raise ValueError(f"Unknown speech act type: {speech_act_type}")

    def __len__(self):
        return len(self.intents)

    def __getitem__(self, idx):
        sample = self.intents[idx]
        return sample


def get_speech_act(intent):
    speech_act_code = intent[:len(SPEECH_ACTS)]
    return SPEECH_ACTS[speech_act_code.nonzero().item()]


def get_speech_act_code(intent):
    return intent[:len(SPEECH_ACTS)]


def get_object(intent):
    return intent[len(SPEECH_ACTS):]


def get_objects(intents):
    return intents[:, len(SPEECH_ACTS):]


class SignalingGameSpeechActsDiscriminationDataset(IterableDataset):

    def __init__(self, num_features, num_values, num_distractors, speech_acts):
        self.label_true = num_distractors
        self.label_false = num_distractors + 1
        self.num_distractors = num_distractors
        self.num_features = num_features
        self.num_values = num_values
        self.speech_acts = speech_acts

    def get_sample(self):
        speech_act = random.choice(self.speech_acts)

        distractors = [generate_object(self.num_features, self.num_values) for _ in range(self.num_distractors)]
        receiver_input = torch.stack(distractors)

        if speech_act == REQUEST:
            target_position = random.choice(range(self.num_distractors))
            label = target_position
            sender_object = receiver_input[target_position]
            sender_input = torch.cat((speech_act_to_one_hot(speech_act), sender_object))
            receiver_input[target_position] = get_object(sender_input)

        elif speech_act == QUESTION_EXISTS:
            question_content = generate_question_content(self.num_features, self.num_values)
            sender_input = torch.cat((speech_act_to_one_hot(speech_act), question_content))

            label = self.label_false
            for object in receiver_input:
                if torch.sum(object * question_content) > 0:
                    label = self.label_true
                    break
        elif speech_act == QUESTION_FORALL:
            question_content = generate_question_content(self.num_features, self.num_values)
            sender_input = torch.cat((speech_act_to_one_hot(speech_act), question_content))

            label = self.label_true
            for object in receiver_input:
                if torch.sum(object * question_content) == 0:
                    label = self.label_false
                    break
        else:
            raise ValueError("Unknown speech act: ", speech_act)

        return sender_input, receiver_input, label

    def __iter__(self):
        while 1:
            yield self.get_sample()