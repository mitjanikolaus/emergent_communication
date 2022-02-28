import torch
import yaml

from data import SignalingGameDataModule
from language_analysis import compute_topsim


def create_compositional_language(dataset, num_features, num_values, message_length, vocab_size):
    assert num_features <= message_length
    assert num_values <= vocab_size

    meanings = []
    messages = []
    for meaning in dataset:
        message = torch.zeros(message_length, dtype=torch.long)
        for feature_idx, feature in enumerate(range(num_features)):
            feature_value = torch.where(meaning[feature_idx * num_values:(feature_idx + 1) * num_values] == 1)[0]
            message[feature_idx] = feature_value
        meanings.append(meaning)
        messages.append(message)

    meanings = torch.stack(meanings)
    messages = torch.stack(messages)
    return meanings, messages


if __name__ == '__main__':
    with open("hparams.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    datamodule = SignalingGameDataModule(num_features=config["model"]["num_features"],
                                        num_values=config["model"]["num_values"], num_distractors=config["data"]["num_distractors"],
                                         batch_size=config["data"]["batch_size"],
                                         num_workers=config["data"]["num_workers"])

    dataset = datamodule.data_train
    meanings, messages = create_compositional_language(dataset, num_features=config["model"]["num_features"], num_values=config["model"]["num_values"], message_length=config["model"]["max_len"], vocab_size=config["model"]["vocab_size"])
    topsim = compute_topsim(meanings, messages)
    print("Topsim: ", topsim)

