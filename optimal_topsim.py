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
        message = meaning.view(num_features, num_values).argmax(dim=-1)
        # Append zeros for full message length:
        message = torch.cat((message, torch.zeros(message_length - len(message))))
        meanings.append(meaning)
        messages.append(message)

    meanings = torch.stack(meanings)
    messages = torch.stack(messages)
    return meanings, messages


if __name__ == '__main__':
    with open("hparams.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    datamodule = SignalingGameDataModule(num_features=config["model"]["num_features"],
                                         num_values=config["model"]["num_values"],
                                         max_num_objects=config["data"]["max_num_objects"],
                                         test_set_size=config["data"]["test_set_size"],
                                         batch_size=config["data"]["batch_size"],
                                         num_workers=config["data"]["num_workers"])

    dataset = datamodule.train_dataset
    meanings, messages = create_compositional_language(dataset, num_features=config["model"]["num_features"], num_values=config["model"]["num_values"], message_length=config["model"]["max_len"], vocab_size=config["model"]["vocab_size"])
    topsim = compute_topsim(meanings, messages)
    print("Topsim: ", topsim)

