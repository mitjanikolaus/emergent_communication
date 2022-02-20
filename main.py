from argparse import Namespace

import yaml
import pytorch_lightning as pl

from data import SignalingGameDataModule
from generate_data import DATA_PATH, generate_data
from model import SignalingGameModule


def run():
    with open("hparams.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generate_data(config["model"]["num_features"], config["model"]["num_values"])

    datamodule = SignalingGameDataModule(DATA_PATH, batch_size=config["data"]["batch_size"])

    # for batch in train_loader:
    #     sender_input, receiver_input, labels = batch
    #     print(sender_input)
    #     print(receiver_input)
    #     print(labels)

    model = SignalingGameModule(**config)

    trainer_args = config["trainer"]
    trainer = pl.Trainer.from_argparse_args(Namespace(**trainer_args))

    trainer.fit(model, datamodule)


if __name__ == '__main__':
    run()

