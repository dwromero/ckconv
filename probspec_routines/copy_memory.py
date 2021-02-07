# torch
import torch

# built-in
import copy
import matplotlib.pyplot as plt

# logging
import wandb

import ckconv


def _train(
    model,
    criterion,
    optimizer,
    dataloaders,
    lr_scheduler,
    config,
    test_loader,
):
    # Training parameters
    epochs = config.epochs
    device = config.device
    clip = config.clip

    log_interval = 50
    logger_step = 1
    n_classes = 10

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy_train = 0.0
    best_accuracy_test = 0.0
    best_loss_test = 999

    # L1-loss
    # sparse_loss = L1Loss(weight_loss=0.0)  # Not useful with Sine Nets

    # Train
    model.train()
    correct = 0
    counter = 0
    for epoch in range(epochs):
        print("Epoch {}/{}".format(epoch + 1, epochs))
        print("-" * 30)
        # Print current learning rate
        for param_group in optimizer.param_groups:
            print("Learning Rate: {}".format(param_group["lr"]))
        print("-" * 30)
        # log learning_rate of the epoch
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=logger_step)

        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloaders["train"]):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x.unsqueeze(1).contiguous())
            loss = criterion(
                out.view(-1, n_classes), y.view(-1)
            )  # + sparse_loss(model)
            pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            counter += out.view(-1, n_classes).size(0)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx > 0 and batch_idx % log_interval == 0:
                avg_loss = total_loss / log_interval
                acc = correct.item() / counter
                print(
                    "| Epoch {:3d} | {:5d}/{:5d} batches |"
                    "loss {:5.8f} | accuracy {:5.4f}".format(
                        epoch + 1,
                        batch_idx,
                        len(dataloaders["train"].dataset) // x.shape[0] + 1,
                        avg_loss,
                        100.0 * acc,
                    )
                )
                print(
                    criterion(out.view(-1, n_classes), y.view(-1)).detach().cpu(),
                    # sparse_loss(model).detach().cpu(),
                )
                # --------------------------
                # log statistics of the loss / accuracy
                wandb.log({"loss_train": avg_loss}, step=logger_step)
                wandb.log({"acc_train": acc}, step=logger_step)
                logger_step += 1
                # Summary of best results
                if acc > best_accuracy_train:
                    best_accuracy_train = acc
                    wandb.run.summary["best_train_accuracy"] = acc
                # --------------------------

                total_loss = 0
                correct = 0
                counter = 0

        # Validation:
        model.eval()

        test_loss = 0
        correct = 0
        counter = 0
        for i, (x, y) in enumerate(dataloaders["validation"]):
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                out = model(x.unsqueeze(1).contiguous())
                test_loss += criterion(out.view(-1, n_classes), y.view(-1))
                _, pred = torch.max(out.view(-1, n_classes), 1)
                correct += pred.eq(y.data.view_as(pred)).cpu().sum()
                counter += out.view(-1, n_classes).size(0)

        acc = correct.item() / counter
        print(
            "\nTest set: Average loss: {:.8f}  |  Accuracy: {:.4f}\n".format(
                test_loss.item(), 100.0 * acc
            )
        )
        # --------------------------
        # log statistics of the loss / accuracy
        wandb.log({"loss_test": test_loss.item()}, step=logger_step)
        wandb.log({"acc_test": acc}, step=logger_step)
        # Summary of best results
        if acc > best_accuracy_test:
            best_accuracy_test = acc
            wandb.run.summary["best_test_accuracy"] = acc
            best_model_wts = copy.deepcopy(model.state_dict())
        elif (
            acc == best_accuracy_test
        ):  # If the performance is the same, save weights wrt. best loss.
            if test_loss < best_loss_test:
                best_loss_test = test_loss
                wandb.run.summary["best_test_loss"] = test_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print(best_loss_test)
        # --------------------------

        # Update scheduler
        if lr_scheduler.__class__.__name__ == "MultiStepLR":
            lr_scheduler.step()
        # Update step in ReduceOnPlateauScheduler
        if lr_scheduler.__class__.__name__ == "ReduceLROnPlateau":
            lr_scheduler.step(test_loss.item())

        print()

    # Report best results
    print("Best Val Acc: {:.4f}".format(best_accuracy_test))
    # Load best model weights
    model.load_state_dict(best_model_wts)

    return model


def _test(model, test_loader, config):
    device = config.device
    model.eval()
    model.to(device)

    correct = 0
    counter = 0
    n_classes = 10

    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            out = model(x.unsqueeze(1).contiguous())
            pred = out.view(-1, n_classes).data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()
            counter += out.view(-1, n_classes).size(0)

    print("\nTest set: Accuracy: {:.4f}\n".format(100.0 * correct / counter))
    return model
