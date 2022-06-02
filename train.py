import argparse
from argparse import Namespace

import yaml
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from data import SignalingGameDataModule
from model import SignalingGameModule


def run(args):
    with open(args.hparams) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    seed_everything(config["seed"], workers=True)

    datamodule = SignalingGameDataModule(speech_acts=config["model"]["speech_acts"],
                                         num_features=config["model"]["num_features"],
                                         num_values=config["model"]["num_values"],
                                         num_objects=config["data"]["num_objects"],
                                         max_num_objects=config["data"]["max_num_objects"],
                                         test_set_size=config["data"]["test_set_size"],
                                         batch_size=config["data"]["batch_size"],
                                         num_workers=config["data"]["num_workers"])

    model = SignalingGameModule(**config)

    trainer = pl.Trainer.from_argparse_args(Namespace(**config["trainer"]))

    # Initial validation
    # trainer.validate(model, datamodule=datamodule, verbose=False)

    # Training
    trainer.fit(model, datamodule)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", type=str, default="hparams.yaml")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run(args)
