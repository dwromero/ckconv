import torch
import models
import ckconv
import wandb


def get_model(config):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """

    in_channels = -1
    if config.dataset in ["AddProblem"]:
        in_channels = 2
    elif config.dataset in ["CopyMemory", "MNIST"]:
        in_channels = 1
    elif config.dataset in ["CIFAR10", "CharTrajectories"]:
        in_channels = 3
    elif config.dataset in ["SpeechCommands"]:
        if config.mfcc:
            in_channels = 20
        else:
            in_channels = 1
    else:
        raise NotImplementedError("Dataset {} not found.".format(config.dataset))

    # Consider the exist_mask channel for irregularly sampled cases.
    if config.drop_rate != 0 and config.dataset in [
        "CharTrajectories",
        "SpeechCommands",
    ]:
        in_channels = in_channels + 1

    model_name = "%s_%s" % (config.dataset, config.model)

    model = {
        "MNIST_BFCNN": lambda: models.seqImg_BFCNN(
            in_channels=in_channels,
            out_channels=10,
            hidden_channels=config.no_hidden,
            kernel_size=28 * 28,
            num_blocks=config.no_blocks,
            bias=True,
            dropout=config.dropout,
            weight_dropout=config.weight_dropout,
            pool=config.pool,
        ),
        "SpeechCommands_BFCNN": lambda: models.seqImg_BFCNN(
            in_channels=in_channels,
            out_channels=10,
            hidden_channels=config.no_hidden,
            kernel_size=161 if config.mfcc else 16000,
            num_blocks=config.no_blocks,
            bias=True,
            dropout=config.dropout,
            weight_dropout=config.weight_dropout,
            pool=config.pool,
        ),
        "AddProblem_TCN": lambda: models.AddProblem_TCN(
            input_size=in_channels,
            output_size=1,
            num_channels=[config.no_hidden] * config.tcn_no_levels,
            kernel_size=config.cnn_kernel_size,
            dropout=config.dropout,
        ),
        "CopyMemory_TCN": lambda: models.CopyMemory_TCN(
            input_size=in_channels,
            output_size=10,
            num_channels=[config.no_hidden] * config.no_blocks,
            kernel_size=config.cnn_kernel_size,
            dropout=config.dropout,
        ),
        "MNIST_TCN": lambda: models.MNIST_TCN(
            input_size=in_channels,
            output_size=10,
            num_channels=[config.no_hidden] * config.no_blocks,
            kernel_size=config.cnn_kernel_size,
            dropout=config.dropout,
        ),
        "AddProblem_CKCNN": lambda: models.AddProblem_CKCNN(
            in_channels=in_channels,
            hidden_channels=config.no_hidden,
            num_blocks=config.no_blocks,
            kernelnet_hidden_channels=config.kernelnet_no_hidden,
            kernelnet_activation_function=config.kernelnet_activation_function,
            kernelnet_norm_type=config.kernelnet_norm_type,
            dim_linear=1,
            bias=True,
            omega_0=config.kernelnet_omega_0,
            dropout=config.dropout,
            weight_dropout=config.weight_dropout,
            pool=config.pool,
        ),
        "CopyMemory_CKCNN": lambda: models.CopyMemory_CKCNN(
            in_channels=in_channels,
            hidden_channels=config.no_hidden,
            num_blocks=config.no_blocks,
            kernelnet_hidden_channels=config.kernelnet_no_hidden,
            kernelnet_activation_function=config.kernelnet_activation_function,
            kernelnet_norm_type=config.kernelnet_norm_type,
            dim_linear=1,
            bias=True,
            omega_0=config.kernelnet_omega_0,
            dropout=config.dropout,
            weight_dropout=config.weight_dropout,
            pool=config.pool,
        ),
        "MNIST_CKCNN": lambda: models.seqImg_CKCNN(
            in_channels=in_channels,
            out_channels=10,
            hidden_channels=config.no_hidden,
            num_blocks=config.no_blocks,
            kernelnet_hidden_channels=config.kernelnet_no_hidden,
            kernelnet_activation_function=config.kernelnet_activation_function,
            kernelnet_norm_type=config.kernelnet_norm_type,
            dim_linear=1,
            bias=True,
            omega_0=config.kernelnet_omega_0,
            dropout=config.dropout,
            weight_dropout=config.weight_dropout,
            pool=config.pool,
        ),
        "CIFAR10_CKCNN": lambda: models.seqImg_CKCNN(
            in_channels=in_channels,
            out_channels=10,
            hidden_channels=config.no_hidden,
            num_blocks=config.no_blocks,
            kernelnet_hidden_channels=config.kernelnet_no_hidden,
            kernelnet_activation_function=config.kernelnet_activation_function,
            kernelnet_norm_type=config.kernelnet_norm_type,
            dim_linear=1,
            bias=True,
            omega_0=config.kernelnet_omega_0,
            dropout=config.dropout,
            weight_dropout=config.weight_dropout,
            pool=config.pool,
        ),
        "SpeechCommands_CKCNN": lambda: models.seqImg_CKCNN(
            in_channels=in_channels,
            out_channels=10,
            hidden_channels=config.no_hidden,
            num_blocks=config.no_blocks,
            kernelnet_hidden_channels=config.kernelnet_no_hidden,
            kernelnet_activation_function=config.kernelnet_activation_function,
            kernelnet_norm_type=config.kernelnet_norm_type,
            dim_linear=1,
            bias=True,
            omega_0=config.kernelnet_omega_0,
            dropout=config.dropout,
            weight_dropout=config.weight_dropout,
            pool=config.pool,
        ),
        "CharTrajectories_CKCNN": lambda: models.seqImg_CKCNN(
            in_channels=in_channels,
            out_channels=20,
            hidden_channels=config.no_hidden,
            num_blocks=config.no_blocks,
            kernelnet_hidden_channels=config.kernelnet_no_hidden,
            kernelnet_activation_function=config.kernelnet_activation_function,
            kernelnet_norm_type=config.kernelnet_norm_type,
            dim_linear=1,
            bias=True,
            omega_0=config.kernelnet_omega_0,
            dropout=config.dropout,
            weight_dropout=config.weight_dropout,
            pool=config.pool,
        ),
    }[model_name]()

    # print number parameters
    print("Number of parameters:", ckconv.utils.num_params(model))
    # wandb.run.summary["no_params"] = ckconv.utils.num_params(model)

    # Check if multi-GPU available and if so, use the available GPU's
    print("GPU's available:", torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = torch.nn.DataParallel(model)  # Required for multi-GPU
    model.to(config.device)
    torch.backends.cudnn.benchmark = True

    return model
