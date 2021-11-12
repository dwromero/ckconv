import torch
from datasets import (
    AdditionProblem,
    CopyMemory,
    MNIST,
    CIFAR10,
    SpeechCommands,
    CharTrajectories,
    PhysioNet,
)

import ml_collections
from typing import Tuple


def dataset_constructor(
    config: ml_collections.ConfigDict,
) -> Tuple[
    torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset
]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple (training_set, validation_set, test_set)
    """
    dataset = {
        "AddProblem": AdditionProblem,
        "CopyMemory": CopyMemory,
        "MNIST": MNIST,
        "CIFAR10": CIFAR10,
        "SpeechCommands": SpeechCommands,
        "CharTrajectories": CharTrajectories,
        "PhysioNet": PhysioNet,
    }[config.dataset]

    training_set = dataset(
        partition="train",
        seq_length=config.seq_length,
        memory_size=config.memory_size,
        mfcc=config.mfcc,
        sr=config.sr_train,
        dropped_rate=config.drop_rate,
    )
    test_set = dataset(
        partition="test",
        seq_length=config.seq_length,
        memory_size=config.memory_size,
        mfcc=config.mfcc,
        sr=config.sr_train
        if config.sr_test == 0
        else config.sr_test,  # Test set can be sample differently.
        dropped_rate=config.drop_rate,
    )
    if config.dataset in ["SpeechCommands", "CharTrajectories", "PhysioNet"]:
        validation_set = dataset(
            partition="val",
            seq_length=config.seq_length,
            memory_size=config.memory_size,
            mfcc=config.mfcc,
            sr=config.sr_train,
            dropped_rate=config.drop_rate,
        )
    else:
        validation_set = None
    return training_set, validation_set, test_set


def get_dataset(
    config: ml_collections.ConfigDict,
    num_workers: int = 4,
    data_root="./data",
) -> Tuple[dict, torch.utils.data.DataLoader]:
    """
    Create datasets loaders for the chosen datasets
    :return: Tuple ( dict(train_loader, val_loader) , test_loader)
    """
    training_set, validation_set, test_set = dataset_constructor(config)

    training_loader = torch.utils.data.DataLoader(
        training_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if validation_set is not None:
        val_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
    else:
        val_loader = test_loader

    dataloaders = {"train": training_loader, "validation": val_loader}

    return dataloaders, test_loader
