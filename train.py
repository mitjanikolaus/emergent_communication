import argparse
import os
import pickle

from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data import SignalingGameDataModule
from model import SignalingGameModule


def run(config):
    seed_everything(config.seed, workers=True)

    checkpoint_callback_1 = ModelCheckpoint(monitor="val_acc", mode="max", save_last=True,
                                          filename="{epoch:02d}-{val_acc:.2f}")
    checkpoint_callback_2 = ModelCheckpoint(monitor="val_acc_no_noise", mode="max", save_last=True,
                                          filename="{epoch:02d}-{val_acc_no_noise:.2f}")
    early_stop_callback = EarlyStopping(monitor="val_acc_no_noise", patience=config.patience, verbose=True, mode="max",
                                        min_delta=0.01, stopping_threshold=0.99)

    datamodule = SignalingGameDataModule(num_attributes=config.num_attributes,
                                         num_values=config.num_values,
                                         max_num_objects=config.max_num_objects,
                                         test_set_size=config.test_set_size,
                                         batch_size=config.batch_size,
                                         num_workers=config.num_workers,
                                         seed=config.seed)

    checkpoint = config.load_checkpoint

    if checkpoint:
        print("Loading checkpoint: "+checkpoint)
        model = SignalingGameModule.load_from_checkpoint(checkpoint, **vars(config))
    else:
        model = SignalingGameModule(**vars(config))

    trainer = Trainer.from_argparse_args(config, callbacks=[checkpoint_callback_1, checkpoint_callback_2, early_stop_callback])

    # Training
    trainer.fit(model, datamodule)

    # Evaluation
    best_model_1 = SignalingGameModule.load_from_checkpoint(checkpoint_callback_1.best_model_path)
    epoch_ckpt_1 = int(checkpoint_callback_1.best_model_path.split("epoch=")[1].split("-")[0])
    best_model_1.force_log = True
    print("\n\nEvaluating: ", checkpoint_callback_1.best_model_path)
    results_1 = trainer.validate(best_model_1, datamodule, verbose=True)
    path_1 = checkpoint_callback_1.best_model_path.replace(".ckpt", "_results.pickle")
    pickle.dump(results_1, open(path_1, "wb"))

    best_model_2 = SignalingGameModule.load_from_checkpoint(checkpoint_callback_2.best_model_path)
    epoch_ckpt_2 = int(checkpoint_callback_2.best_model_path.split("epoch=")[1].split("-")[0])
    if epoch_ckpt_1 != epoch_ckpt_2:
        best_model_2.force_log = True
        print("\n\nEvaluating: ", checkpoint_callback_2.best_model_path)
        results_2 = trainer.validate(best_model_2, datamodule, verbose=True)
    else:
        results_2 = results_1
    path_2 = checkpoint_callback_2.best_model_path.replace(".ckpt", "_results.pickle")
    pickle.dump(results_2, open(path_2, "wb"))


def get_args():
    parser = argparse.ArgumentParser()

    # add model specific args
    parser = SignalingGameModule.add_model_specific_args(parser)

    # add all the available trainer options to argparse
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(max_epochs=-1,
                        check_val_every_n_epoch=100,
                        log_every_steps=1000,
                        num_sanity_val_steps=3,
                        max_time="00:46:00:00", # Default max time: 46 hours
                        )

    # add general and data args
    parser.add_argument("--seed", type=int, default="1")
    parser.add_argument("--max-num-objects", type=int, default="100000")
    parser.add_argument("--batch-size", type=int, default="5120")
    parser.add_argument("--test-set-size", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default="0")
    parser.add_argument("--patience", type=int, default=50)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    run(args)
