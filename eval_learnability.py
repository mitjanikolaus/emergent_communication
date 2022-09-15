import argparse
from argparse import Namespace

import torch.cuda
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from data import SignalingGameDataModule, REQUEST
from model import SignalingGameModule


def run(args):
    model = SignalingGameModule.load_from_checkpoint(args.checkpoint)
    config = model.hparams
    #
    if not torch.cuda.is_available():
        print("CUDA not available, adjusting loaded config file.")
        config['trainer']['gpus'] = 0
    # seed_everything(config["seed"], workers=True)

    config['trainer']['log_every_n_steps'] = 10
    config['trainer']['val_check_interval'] = 100
    model.params['log_topsim_on_validation'] = False

    # model.model_hparams['receiver_aux_loss'] = False

    speech_acts_used = [REQUEST]

    datamodule = SignalingGameDataModule(speech_acts=config["model"]["speech_acts"],
                                         num_features=config["model"]["num_features"],
                                         num_values=config["model"]["num_values"],
                                         num_objects=config["data"]["num_objects"],
                                         max_num_objects=config["data"]["max_num_objects"],
                                         test_set_size=config["data"]["test_set_size"],
                                         batch_size=config["data"]["batch_size"],
                                         num_workers=config["data"]["num_workers"],
                                         speech_acts_used=speech_acts_used)

    # Reset receivers
    if args.mlp_receivers:
        model.init_MLP_receivers()
        model.params['receiver_aux_loss'] = False
        model.params['receiver_embed_dim'] = 32
        # model.model_hparams['receiver_hidden_dim'] = 32

    # Freeze senders
    model.freeze_senders()

    trainer = pl.Trainer.from_argparse_args(Namespace(**config["trainer"]))

    # Training
    trainer.fit(model, datamodule)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--mlp-receivers", action="store_true", default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run(args)

