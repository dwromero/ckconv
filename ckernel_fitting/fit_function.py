"""
In this experiments, we analyze the fitting properties of the CKConv parameterization.
In particular we ask the following question:

 Which kind of functions are we able to fit via continuous kernels?
"""
# general
import os
import wandb
import ml_collections
import sys
import matplotlib.pyplot as plt
import copy

# torch
import numpy as np
import torch

# args
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

# project
import ckconv.nn
from ckernel_fitting.functions import get_function_to_fit


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
    wandb.init(
        project="ckconv",
        config=copy.deepcopy(dict(config)),
        group="kernelfit_{}".format(config.function),
        entity="vu_uva_team",
        tags=["kernelfit", config.function],
        # save_code=True,
        # job_type=config.function,
    )

    # Load the model: The model is always equal to a continuous kernel
    model = get_model(config)

    # get function to fit
    f = get_function_to_fit(config)
    if config.padding != 0:
        f = np.pad(f, (0, config.padding), "constant", constant_values=(0, 0))
    # plot function to fit
    plot_function_to_fit(f, config)

    # input to the model
    x = (
        torch.from_numpy(np.linspace(-1.0, 0.0, len(f)))
        .type(torch.FloatTensor)
        .to(config.device)
        .unsqueeze(0)
        .unsqueeze(0)
    )
    f = (
        torch.from_numpy(f)
        .type(torch.FloatTensor)
        .to(config.device)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # fit the kernel
    # --------------
    log_interval = 50
    # Define optimizer
    lr = config.lr
    optimizer = getattr(torch.optim, config.optim)(model.parameters(), lr=lr)

    iter = 1
    total_loss = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy_train = 999

    for iterations in range(config.no_iterations):
        model.train()
        optimizer.zero_grad()

        output = model(x)
        loss = torch.nn.functional.mse_loss(output, f)
        loss.backward()

        # plot initial values of the kernel
        if iter == 1:
            plot_init_value_kernel(output.detach().cpu().squeeze().numpy())

        optimizer.step()
        iter += 1
        total_loss += loss.item()
        wandb.log({"reconstruction_loss": loss.item()}, step=iter)

        if iter % log_interval == 0:
            cur_loss = total_loss / log_interval
            print(
                "Iter: {:2d}/{:6d} \tLearning rate: {:.4f}\tLoss: {:.6f}".format(
                    iter, config.no_iterations, lr, cur_loss
                )
            )
            total_loss = 0

        if loss.item() < best_accuracy_train:
            best_accuracy_train = loss.item()
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)

    # Save the model
    torch.save(model.module.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
    # --------------

    # Check the fitting
    # -----------------
    model.eval()
    with torch.no_grad():
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, f)
        # log
        wandb.run.summary["reconstruction_loss"] = loss
        print("\nTest: loss: {}\n".format(loss.item()))
        # plot results and log them
        plot_fitted_kernel(
            output.detach().cpu().squeeze().numpy(),
            f.detach().cpu().squeeze().numpy(),
            loss,
            config,
        )
    # -----------------


def get_model(config):
    # Load the model: The model is always equal to a continuous kernel
    if config.kernelnet_type == "SIREN":
        model_type = ckconv.nn.KernelNet
    elif config.kernelnet_type == "RFNet":
        model_type = ckconv.nn.RFNet
    else:
        raise NotImplementedError(f"{config.kernelnet_type}")

    model = model_type(
        in_channels=1,
        out_channels=1,
        hidden_channels=config.kernelnet_no_hidden,
        activation_function=config.kernelnet_activation_function,
        norm_type=config.kernelnet_norm_type,
        dim_linear=1,
        bias=True,
        omega_0=config.kernelnet_omega_0,
        weight_dropout=0,
    )
    # Set device
    config.device = (
        "cuda:0" if (config.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    model = torch.nn.DataParallel(model)  # Required for multi-GPU
    model.to(config.device)
    torch.backends.cudnn.benchmark = True

    return model


def plot_function_to_fit(f, config):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(f)
    ax.set_xticks([])
    plt.title("Ground Truth")
    plt.tight_layout()
    # plt.savefig("{}.png".format(config.function), dpi=300)
    wandb.log({"ground_truth": wandb.Image(plt)})
    # plt.show()


def plot_input_kernel(x):
    plt.figure()
    plt.plot(x)
    plt.xticks([])
    plt.title("Input of the kernel (relative positions)")
    plt.tight_layout()
    plt.show()


def plot_init_value_kernel(output):
    plt.figure(dpi=300)
    plt.plot(output)
    plt.title("Initial value of the kernel")
    plt.xticks([])
    plt.tight_layout()
    wandb.log({"initial_kernel": wandb.Image(plt)})
    # plt.show()


def plot_fitted_kernel(output, f, loss, config):
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    x_values = np.linspace(config.min, 0, f.shape[-1])
    ax.plot(x_values, f, label="function")
    ax.plot(x_values, output, label="fitted kernel")
    ax.set_xticks([])
    ax.text(
        0.99,
        0.013,
        "Loss: {:.3e}".format(loss.item()),
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=ax.transAxes,
        color="Black",
        fontsize=12,
        weight="roman",
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.9, "pad": 4},
    )
    ax.legend(loc=1)
    plt.title("Comparison function and fitted kernel. Loss: {:.4e}".format(loss.item()))
    plt.tight_layout()
    # plt.savefig(
    #     "{}_{}_{}.png".format(
    #         config.function,
    #         config.kernelnet_activation_function,
    #         config.comment,
    #     ),
    #     dpi=300,
    # )
    wandb.log({"fitted_kernel": wandb.Image(plt)})
    # plt.show()

    plt.figure(dpi=300)
    plt.plot(x_values, f - output)
    plt.xticks([])
    plt.title("Difference (f - output). Loss: {:.4e}".format(loss.item()))
    plt.tight_layout()
    wandb.log({"diff_fit_gt": wandb.Image(plt)})
    # plt.show()


if __name__ == "__main__":
    app.run(main)
