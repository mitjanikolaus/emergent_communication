from argparse import Namespace

import yaml
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from data import SignalingGameDiscriminationDataset
from generate_data import DATA_PATH, generate_data
from model import SignalingGameModule


def run():
    with open("hparams.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generate_data(config["model"]["num_features"], config["model"]["num_values"])

    dataset = SignalingGameDiscriminationDataset(DATA_PATH)
    train_loader = DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=True)

    # for batch in train_loader:
    #     sender_input, receiver_input, labels = batch
    #     print(sender_input)
    #     print(receiver_input)
    #     print(labels)

    model = SignalingGameModule(**config)

    trainer = pl.Trainer.from_argparse_args(Namespace(**config["trainer"]))

    trainer.fit(model, train_loader)


if __name__ == '__main__':
    run()

