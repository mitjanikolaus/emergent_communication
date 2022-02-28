from argparse import Namespace

import yaml
import pytorch_lightning as pl

from data import SignalingGameDataModule, SignalingGameDataset
from model import SignalingGameModule


def run():
    with open("hparams.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    datamodule = SignalingGameDataModule(num_features=config["model"]["num_features"],
                                        num_values=config["model"]["num_values"], num_distractors=config["data"]["num_distractors"],
                                         batch_size=config["data"]["batch_size"],
                                         num_workers=config["data"]["num_workers"])

    model = SignalingGameModule(**config)

    trainer_args = config["trainer"]
    trainer = pl.Trainer.from_argparse_args(Namespace(**trainer_args))

    # Initial validation
    trainer.validate(model, datamodule=datamodule, verbose=False)

    # Training
    trainer.fit(model, datamodule)


if __name__ == '__main__':
    run()

