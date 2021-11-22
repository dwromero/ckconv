# torch
import torch

# built-in
import copy
import os
import datetime
import math
import numpy as np

# logging
import wandb

# project
import probspec_routines as ps_routines
from tester import test
import ckconv

import sklearn


def train(model, dataloaders, config, test_loader):

    criterion = {
        "AddProblem": torch.nn.functional.mse_loss,
        "CopyMemory": torch.nn.CrossEntropyLoss(),
        "MNIST": torch.nn.CrossEntropyLoss(),
        "CIFAR10": torch.nn.CrossEntropyLoss(),
        "SpeechCommands": torch.nn.CrossEntropyLoss(),
        "CharTrajectories": torch.nn.CrossEntropyLoss(),
        "PhysioNet": torch.nn.BCEWithLogitsLoss(),
        "PennTreeBankChar": torch.nn.CrossEntropyLoss(),
    }[config.dataset]

    train_function = {
        "AddProblem": ps_routines.add_problem_train,
        "CopyMemory": ps_routines.copy_problem_train,
        "MNIST": _train_classif,
        "CIFAR10": _train_classif,
        "SpeechCommands": _train_classif,
        "CharTrajectories": _train_classif,
        "PhysioNet": _train_classif,
        "PennTreeBank": _train_language_modeling,
        "PennTreeBankChar": _train_language_modeling,
    }[config.dataset]

    # Define optimizer and scheduler
    optimizer = get_optimizer(model.parameters(), config)
    lr_scheduler = get_scheduler(optimizer, config)

    # train network
    _ = train_function(
        model,
        criterion,
        optimizer,
        dataloaders,
        lr_scheduler,
        config,
        test_loader,
    )
    # save model and log it
    torch.save(model.state_dict(), config.path)
    # Save the model in the exchangeable ONNX format
    torch.save(model.module.state_dict(), config.path)
    torch.save(model.module.state_dict(), os.path.join(wandb.run.dir, "model.pt"))
    torch.save(
        model.module.state_dict(),
        os.path.join(wandb.run.dir, config.path.split("/")[-1]),
    )


def get_optimizer(model_parameters, config):
    """
    Create an optimizer for a given model
    :param model_parameters: a list of parameters to be trained
    :return: optimizer
    """
    if config.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            model_parameters,
            lr=config.lr,
            momentum=config.optimizer_momentum,
            # weight_decay=config.weight_decay,
        )
    elif config.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            model_parameters,
            lr=config.lr,
            # weight_decay=config.weight_decay,
        )
    elif config.optimizer == "RMSprop":
        optimizer = torch.optim.RMSprop(
            model_parameters,
            lr=config.lr,
            # weight_decay=config.weight_decay,
        )
    else:
        raise ValueError("Unexpected value for optimizer")

    return optimizer


def get_scheduler(optimizer, config):
    """
    Creates a learning rate scheduler for a given model
    :param optimizer: the optimizer to be used
    :return: scheduler
    """
    if config.scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.sched_decay_steps,
            gamma=1.0 / config.sched_decay_factor,
        )
    elif config.scheduler == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=1.0 / config.sched_decay_factor,
            patience=config.sched_patience,
            verbose=True,
            # threshold_mode="rel",
            # min_lr=2.5e-4,
        )
    else:
        lr_scheduler = None
        print("WARNING! The scheduler is not recognized. No scheduler will be used.")
    return lr_scheduler


def _train_classif(
        model, criterion, optimizer, dataloader, lr_scheduler, config, test_loader
):
    weight_regularizer = ckconv.nn.LnLoss(weight_loss=config.weight_decay, norm_type=2)
    # Training parameters
    epochs = config.epochs
    device = config.device
    # clip = config.clip
    if config.dataset == "MNIST" and config.permuted:
        permutation = torch.Tensor(np.random.permutation(784).astype(np.float64)).long()

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 999

    # Counter for epochs without improvement
    epochs_no_improvement = 0
    max_epochs_no_improvement = config.max_epochs_no_improvement

    # iterate over epochs
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print("Learning Rate: {}".format(param_group["lr"]))
        print("-" * 30)
        # log learning_rate of the epoch
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)

        # Each epoch consist of training and validation
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # Accumulate accuracy and loss
            running_loss = 0
            running_corrects = 0
            total = 0

            true_y_cpus = []
            pred_y_cpus = []

            # iterate over data
            for inputs, labels in dataloader[phase]:

                if config.dataset in ["MNIST", "CIFAR10"]:
                    _, in_channels, x, y = inputs.shape
                    inputs = inputs.view(-1, in_channels, x * y)
                if config.permuted and config.dataset == "MNIST":
                    inputs = inputs[:, :, permutation]

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(train):
                    # FwrdPhase:
                    inputs = torch.dropout(inputs, config.dropout_in, train)
                    outputs = model(inputs)

                    if len(outputs.shape) == 1:
                        labels = labels.float()
                        preds = (outputs > 0.0).int()
                    else:
                        _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)
                    # Regularization:
                    if config.weight_decay != 0.0:
                        loss = loss + weight_regularizer(model)

                    # Save for AUC
                    if config.report_auc:
                        true_y_cpus.append(labels.detach().cpu())
                        pred_y_cpus.append(outputs.detach().cpu())

                    # BwrdPhase:
                    if phase == "train":
                        loss.backward()
                        if config.clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), config.clip
                            )
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total += labels.size(0)

            # statistics of the epoch
            epoch_loss = running_loss / total
            epoch_acc = running_corrects / total
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            print(datetime.datetime.now())

            # log statistics of the epoch
            wandb.log(
                {"accuracy" + "_" + phase: epoch_acc, "loss" + "_" + phase: epoch_loss},
                step=epoch + 1,
            )

            if config.report_auc:
                true_y_cpus = torch.cat(true_y_cpus, dim=0)
                pred_y_cpus = torch.cat(pred_y_cpus, dim=0)

                epoch_auc = sklearn.metrics.roc_auc_score(true_y_cpus, pred_y_cpus)

                print(f"AUC: {epoch_auc}")

                wandb.log(
                    {f"auc_{phase}": epoch_auc},
                    step=epoch + 1,
                )

            # If better validation accuracy, replace best weights and compute the test performance
            if phase == "validation" and epoch_acc >= best_acc:

                # Updates to the weights will not happen if the accuracy is equal but loss does not diminish
                if (epoch_acc == best_acc) and (epoch_loss > best_loss):
                    pass
                else:
                    best_acc = epoch_acc
                    best_loss = epoch_loss

                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Log best results so far and the weights of the model.
                    wandb.run.summary["best_val_accuracy"] = best_acc
                    wandb.run.summary["best_val_loss"] = best_loss

                    # Clean CUDA Memory
                    del inputs, outputs, labels
                    torch.cuda.empty_cache()
                    # Perform test and log results
                    if config.dataset in [
                        "SpeechCommands",
                        "CharTrajectories",
                        "PhysioNet",
                    ]:
                        test_acc, test_auc = test(model, test_loader, config)
                    else:
                        test_acc = best_acc
                    wandb.run.summary["best_test_accuracy"] = test_acc
                    wandb.log({"accuracy_test": test_acc}, step=epoch + 1)

                    if config.report_auc:
                        wandb.run.summary["best_val_auc"] = epoch_auc
                        wandb.run.summary["best_test_auc"] = test_auc
                        wandb.log({"test_auc": test_auc}, step=epoch + 1)

                    # Reset counter of epochs without progress
                    epochs_no_improvement = 0

            elif phase == "validation" and epoch_acc < best_acc:
                # Otherwise, increase counter
                epochs_no_improvement += 1

            # Update scheduler
            if (
                    isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                    and phase == "validation"
            ):
                lr_scheduler.step(epoch_acc)

        # Update scheduler
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.MultiStepLR):
            lr_scheduler.step()
        print()

        #  Check how many epochs without improvement have passed, and, if required, stop training.
        if epochs_no_improvement == max_epochs_no_improvement:
            print(
                f"Stopping training due to {epochs_no_improvement} epochs of no improvement in validation accuracy."
            )
            break

    # Report best results
    print("Best Val Acc: {:.4f}".format(best_acc))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    # Return model and histories
    return model


def _train_language_modeling(
        model, criterion, optimizer, dataloader, lr_scheduler, config, test_loader
):
    print(f'Vocabulary size: {config.vocab_size} \n')
    weight_regularizer = ckconv.nn.LnLoss(weight_loss=config.weight_decay, norm_type=2)
    # Training parameters
    epochs = config.epochs
    device = config.device
    eff_history = config.seq_length - config.valid_seq_len

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 999

    # Counter for epochs without improvement
    epochs_no_improvement = 0
    max_epochs_no_improvement = config.max_epochs_no_improvement

    # compute num params
    # total_params = sum(p.numel() for p in model.parameters())
    # if config.tied_weights:
    #     total_params -= sum(p.numel() for p in model.module.encoder.parameters())
    # wandb.run.summary["num_param"] = total_params
    # iterate over epochs
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print("Learning Rate: {}".format(param_group["lr"]))
        print("-" * 30)
        # log learning_rate of the epoch
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=epoch + 1)

        # Each epoch consist of training and validation
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # Accumulate loss
            running_loss = 0
            total = 0
            running_ppl = 0.
            ppl_N = 0

            # iterate over data
            for inputs, labels in dataloader[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)[:, eff_history:].contiguous().view(-1)

                optimizer.zero_grad()
                train = phase == "train"
                with torch.set_grad_enabled(train):
                    # FwrdPhase:
                    outputs, emb = model(inputs, return_emb=True)
                    outputs = outputs[:, eff_history:].contiguous().view(-1, config.vocab_size)
                    loss = criterion(outputs, labels)

                    if config.report_ppl or config.report_bpc:
                        n = inputs.shape[1] - eff_history
                        running_ppl += n * loss.item()
                        ppl_N += n

                    # statistics
                    running_loss += loss.item() * labels.shape[0]
                    total += labels.shape[0]
                    # Regularization:
                    if config.weight_decay != 0.0:
                        loss = loss + weight_regularizer(model)

                    # BwrdPhase:
                    if phase == "train":
                        loss.backward()
                        if config.clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), config.clip
                            )
                        optimizer.step()

            # statistics of the epoch
            epoch_loss = running_loss / total
            print("{} Loss: {:.2f}".format(phase, epoch_loss))

            # log statistics of the epoch
            wandb.log(
                {"loss" + "_" + phase: epoch_loss},
                step=epoch + 1,
            )

            if config.report_ppl:
                epoch_ppl = math.exp(running_ppl / ppl_N)
                print(f"PPL: {epoch_ppl:.2f}")
                wandb.log(
                    {f"ppl_{phase}": epoch_ppl},
                    step=epoch + 1,
                )
            if config.report_bpc:
                epoch_bpc = (running_ppl / ppl_N) / math.log(2)
                print(f"BPC: {epoch_bpc:.2f}")
                wandb.log(
                    {f"bpc_{phase}": epoch_bpc},
                    step=epoch + 1,
                )
            # If better validation accuracy, replace best weights and compute the test performance
            if phase == "validation" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                # Log best results so far and the weights of the model.
                wandb.run.summary["best_val_loss"] = best_loss

                # Clean CUDA Memory
                del inputs, outputs, labels
                torch.cuda.empty_cache()
                # Perform test and log results
                test_loss, test_ppl, test_bpc = test(model, test_loader, config)

                if config.report_ppl:
                    wandb.run.summary["best_val_ppl"] = epoch_ppl
                    wandb.run.summary["best_test_ppl"] = test_ppl
                    wandb.log({"test_ppl": test_ppl}, step=epoch + 1)
                if config.report_bpc:
                    wandb.run.summary["best_val_bpc"] = epoch_bpc
                    wandb.run.summary["best_test_bpc"] = test_bpc
                    wandb.log({"test_bpc": test_bpc}, step=epoch + 1)

                # Reset counter of epochs without progress
                epochs_no_improvement = 0

            elif phase == "validation" and epoch_loss >= best_loss:
                # Otherwise, increase counter
                epochs_no_improvement += 1

            # Update scheduler
            if (
                    isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                    and phase == "validation"
            ):
                lr_scheduler.step(-epoch_loss)

        # Update scheduler
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.MultiStepLR):
            lr_scheduler.step()
        print()

        #  Check how many epochs without improvement have passed, and, if required, stop training.
        if epochs_no_improvement == max_epochs_no_improvement:
            print(
                f"Stopping training due to {epochs_no_improvement} epochs of no improvement in validation accuracy."
            )
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    # Return model and histories
    return model
