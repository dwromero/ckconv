import torch
import torch.nn.functional as F
import numpy as np

import wandb

# project
import probspec_routines as ps_routines


def test(model, test_loader, config):

    test_function = {
        "AddProblem": ps_routines.add_problem_test,
        "CopyMemory": ps_routines.copy_problem_test,
        "MNIST": _test_classif,
        "CIFAR10": _test_classif,
        "SpeechCommands": _test_classif,
        "CharTrajectories": _test_classif,
    }[config.dataset]

    test_acc = test_function(model, test_loader, config)
    return test_acc


def _test_classif(model, test_loader, config):
    # send model to device
    device = config.device
    if config.dataset == "MNIST" and config.permuted:
        permutation = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()

    model.eval()
    model.to(device)

    # Summarize results
    correct = 0
    total = 0

    with torch.no_grad():
        # Iterate through data
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if config.dataset in ["MNIST", "CIFAR10"]:
                _, in_channels, x, y = inputs.shape
                inputs = inputs.view(-1, in_channels, x * y)
                if config.permuted and config.dataset == "MNIST":
                    inputs = inputs[:, :, permutation]

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print results
    test_acc = correct / total
    print(
        "Accuracy of the network on the {} test samples: {}".format(
            total, (100 * test_acc)
        )
    )
    return test_acc
