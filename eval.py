import argparse
from argparse import Namespace

import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from data import SignalingGameDataModule
from model import SignalingGameModule


def run(args):
    with open(args.hparams) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    seed_everything(config["seed"], workers=True)

    checkpoint = args.checkpoint
    print("Loading checkpoint: " + checkpoint)
    model = SignalingGameModule.load_from_checkpoint(checkpoint, **config)

    model.params.log_topsim_on_validation = True
    model.params.log_posdis_on_validation = True
    model.params.log_bosdis_on_validation = True
    model.params.log_entropy_on_validation = True

    datamodule = SignalingGameDataModule(num_attributes=config["model"]["num_attributes"],
                                         num_values=config["model"]["num_values"],
                                         max_num_objects=config["data"]["max_num_objects"],
                                         test_set_size=config["data"]["test_set_size"],
                                         batch_size=config["data"]["batch_size"],
                                         num_workers=config["data"]["num_workers"])

    if torch.cuda.is_available():
        config["trainer"]["gpus"] = 1
    else:
        config["trainer"]["gpus"] = 0

    trainer = pl.Trainer.from_argparse_args(Namespace(**config["trainer"]))

    trainer.validate(model, datamodule=datamodule, verbose=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hparams", type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run(args)

