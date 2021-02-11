# general
import os
import wandb
import ml_collections
import sys
import copy

# torch
import numpy as np
import torch

# project
from path_handler import model_path
from model import get_model
import dataset
import trainer
import tester

# args
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", default="config.py")


def main(_):

    if "absl.logging" in sys.modules:
        import absl.logging

        absl.logging.set_verbosity("info")
        absl.logging.set_stderrthreshold("info")

    config = FLAGS.config
    print(config)

    # Set the seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # initialize weight and bias
    os.environ["WANDB_API_KEY"] = "3fe624d6a1979f80f1277200966d17bed042ec31"  ## Place here your API key.
    if not config.train:
        os.environ["WANDB_MODE"] = "dryrun"
    tags = [
        config.model,
        config.dataset,
        config.kernelnet_activation_function,
        "seq_length={}".format(config.seq_length),
    ]
    if config.dataset == "MNIST":
        tags.append(str(config.permuted))

    wandb.init(
        project="ckconv",
        config=copy.deepcopy(dict(config)),
        group=config.dataset,
        # entity="", # Select entity if working in project with other people.
        tags=tags,
        # save_code=True,
        # job_type=config.function,
    )

    # Define the device to be used and move model to that device
    config["device"] = (
        "cuda:0" if (config.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    model = get_model(config)

    # Define transforms and create dataloaders
    dataloaders, test_loader = dataset.get_dataset(config, num_workers=4)

    # WandB – wandb.watch() automatically fetches all layer dimensions, gradients, model parameters and logs them automatically to your dashboard.
    # Using log="all" log histograms of parameter values in addition to gradients
    # wandb.watch(model, log="all", log_freq=200) # -> There was a wandb bug that made runs in Sweeps crash

    # Create model directory and instantiate config.path
    model_path(config)

    if config.pretrained:
        # Load model state dict
        model.module.load_state_dict(torch.load(config.path), strict=False)

    # Train the model
    if config.train:
        # Print arguments (Sanity check)
        print(config)
        # Train the model
        import datetime

        print(datetime.datetime.now())
        trainer.train(model, dataloaders, config, test_loader)

    # Select test function
    tester.test(model, test_loader, config)


if __name__ == "__main__":
    app.run(main)
