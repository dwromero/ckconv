import os
import pathlib


def model_path(config, root="./saved"):

    root = pathlib.Path(root)
    filename = "{}".format(config.dataset)

    # Dataset-specific keys
    if config.dataset in ["AddProblem", "CopyMemory"]:
        filename += "_seqlen_{}".format(
            config.seq_length,
        )
        if config.dataset in ["CopyMemory"]:
            filename += "_memsize_{}".format(
                config.memory_size,
            )

    elif config.dataset in ["MNIST"]:
        filename += "_perm_{}".format(
            config.permuted,
        )

    elif config.dataset in ["CharTrajectories", "SpeechCommands"]:
        if config.dataset in ["SpeechCommands"]:
            filename += "_mfcc_{}".format(
                config.mfcc,
            )
        if (
            config.dataset in ["SpeechCommands"] and not config.mfcc
        ) or config.dataset in ["CharTrajectories"]:
            filename += "_srtr_{}_drop_{}".format(
                config.sr_train,
                config.drop_rate,
            )

    # Model-specific keys
    filename += "_model_{}_blcks_{}_nohid_{}".format(
        config.model,
        config.no_blocks,
        config.no_hidden,
    )
    if config.model == "CKCNN":
        filename += "_kernnohid_{}_kernact_{}".format(
            config.kernelnet_no_hidden,
            config.kernelnet_activation_function,
        )
        if config.kernelnet_activation_function == "Sine":
            filename += "_kernomega0_{}".format(round(config.kernelnet_omega_0, 2))
        else:
            filename += "_kernnorm_{}".format(config.kernelnet_norm_type)

    elif config.model in ["BFCNN", "TCN"]:
        filename += "_kernsize_{}".format(config.cnn_kernel_size)

    # Optimization arguments
    filename += "_bs_{}_optim_{}_lr_{}_ep_{}_dpin_{}_dp_{}_wd_{}_seed_{}_sched_{}_schdec_{}".format(
        config.batch_size,
        config.optimizer,
        config.lr,
        config.epochs,
        config.dropout_in,
        config.dropout,
        config.weight_decay,
        config.seed,
        config.scheduler,
        config.sched_decay_factor,
    )
    if config.scheduler == "plateau":
        filename += "_pat_{}".format(config.sched_patience)
    else:
        filename += "_schsteps_{}".format(config.sched_decay_steps)

    # Comment
    if config.comment != "":
        filename += "_comment_{}".format(config.comment)

    # Add correct termination
    filename += ".pt"

    # Check if directory exists and warn the user if the it exists and train is used.
    os.makedirs(root, exist_ok=True)
    path = root / filename
    config.path = str(path)

    if config.train and path.exists():
        print("WARNING! The model exists in directory and will be overwritten")
