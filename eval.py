import argparse
import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from data import SignalingGameDataModule
from model import SignalingGameModule


def find_best_checkpoint(run_folder):
    checkpoints_folder = os.path.join(run_folder, "checkpoints")

    ckpt_files = os.listdir(checkpoints_folder)
    best_checkpoints = [ckpt for ckpt in ckpt_files if not ckpt == "last.ckpt"]
    # TODO for now returning the first checkpoint found
    # accs = [int(filename[6:-5]) for filename in ckpt_files]
    # print(max(epochs))
    if len(best_checkpoints) == 0:
        raise FileNotFoundError
    assert len(best_checkpoints) == 1
    return os.path.join(checkpoints_folder, best_checkpoints[-1])


def run(args):
    checkpoint_path = find_best_checkpoint(args.run_dir)
    print("Loading checkpoint: " + checkpoint_path)
    model = SignalingGameModule.load_from_checkpoint(checkpoint_path)
    config = model.params

    seed_everything(config["seed"], workers=True)

    datamodule = SignalingGameDataModule(num_attributes=config.num_attributes,
                                         num_values=config.num_values,
                                         max_num_objects=config.max_num_objects,
                                         test_set_size=config.test_set_size,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_workers,
                                         seed=config.seed)

    if torch.cuda.is_available():
        config["gpus"] = 1
    else:
        config["gpus"] = 0

    trainer = pl.Trainer.from_argparse_args(config)

    trainer.validate(model, datamodule=datamodule, verbose=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run(args)

