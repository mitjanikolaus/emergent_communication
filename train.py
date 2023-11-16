import argparse
import pickle

import torch.cuda
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data import SignalingGameDataModule
from model import SignalingGameModule


def run(config):
    seed_everything(config.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')

    checkpoint_callback_1 = ModelCheckpoint(monitor="val_acc", mode="max", save_last=True,
                                          filename="{epoch:02d}-{val_acc:.2f}")
    early_stop_callback = EarlyStopping(monitor="val_acc", patience=config.patience, verbose=True, mode="max",
                                        min_delta=0.01, stopping_threshold=config.stopping_threshold)

    datamodule = SignalingGameDataModule(num_attributes=config.num_attributes,
                                         num_values=config.num_values,
                                         max_num_objects=config.max_num_objects,
                                         val_set_size=config.val_set_size,
                                         test_set_size=config.test_set_size,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_workers,
                                         seed=config.seed,
                                         num_objects=config.discrimination_num_objects,
                                         hard_distractors=config.hard_distractors,
                                         guesswhat=config.guesswhat,
                                         imagenet=config.imagenet,)

    checkpoint = config.load_checkpoint

    if checkpoint:
        print("Loading checkpoint: "+checkpoint)
        model = SignalingGameModule.load_from_checkpoint(checkpoint, **vars(config))
    else:
        model = SignalingGameModule(**vars(config))

    trainer = Trainer.from_argparse_args(config, callbacks=[checkpoint_callback_1, early_stop_callback])

    # Training
    trainer.fit(model, datamodule)

    # Evaluation
    best_model_1 = SignalingGameModule.load_from_checkpoint(checkpoint_callback_1.best_model_path)
    best_model_1.force_log = True
    print("\n\nEvaluating: ", checkpoint_callback_1.best_model_path)
    results_1 = trainer.validate(best_model_1, datamodule, verbose=True)
    path_1 = checkpoint_callback_1.best_model_path.replace(".ckpt", "_results.pickle")
    pickle.dump(results_1, open(path_1, "wb"))


def get_args():
    parser = argparse.ArgumentParser()

    # add model specific args
    parser = SignalingGameModule.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=-1,
                        log_every_steps=10000,
                        check_val_every_n_epoch=20,
                        num_sanity_val_steps=3,
                        limit_val_batches=100,
                        max_time="00:46:00:00", # Default max time: 46 hours
                        )

    # Add general and data args:
    # These args have to be added manually as they're not part of the model args
    parser.add_argument("--seed", type=int, default="1")
    parser.add_argument("--max-num-objects", type=int, default="100000")
    parser.add_argument("--batch-size", type=int, default="1000")
    parser.add_argument("--val-set-size", type=float, default=0.1)
    parser.add_argument("--test-set-size", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default="0")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--stopping-threshold", type=float, default=0.99)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    run(args)
