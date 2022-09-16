import argparse
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data import SignalingGameDataModule
from model import SignalingGameModule


def run(config):
    seed_everything(config.seed, workers=True)

    checkpoint_callback = ModelCheckpoint(monitor="test_acc", mode="max", save_last=True)

    datamodule = SignalingGameDataModule(num_features=config.num_features,
                                         num_values=config.num_values,
                                         max_num_objects=config.max_num_objects,
                                         test_set_size=config.test_set_size,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_workers)

    checkpoint = config.load_checkpoint

    if checkpoint:
        print("Loading checkpoint: "+checkpoint)
        model = SignalingGameModule.load_from_checkpoint(checkpoint, **vars(config))
    else:
        model = SignalingGameModule(**vars(config))

    trainer = Trainer.from_argparse_args(config, callbacks=[checkpoint_callback])

    # Initial validation
    # trainer.validate(model, datamodule=datamodule, verbose=False)

    # Training
    trainer.fit(model, datamodule)


def get_args():
    parser = argparse.ArgumentParser()

    # add model specific args
    parser = SignalingGameModule.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=-1,
                        check_val_every_n_epoch=100,
                        log_every_steps=1000,
                        num_sanity_val_steps=3
                        )

    # add general and data args
    parser.add_argument("--seed", type=int, default="1")
    parser.add_argument("--max-num-objects", type=int, default="100000")
    parser.add_argument("--batch-size", type=int, default="5120")
    parser.add_argument("--test-set-size", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default="0")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    run(args)
