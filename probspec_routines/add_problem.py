# torch
import torch
import torch.nn.functional as F

# built-in
import copy

# logging
import wandb


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
    # clip = config.clip

    log_interval = 100
    logger_step = 1

    # Save best performing weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss_train = 999
    best_loss_test = 999

    # Train
    model.train()

    for epoch in range(epochs):
        # log learning_rate of the epoch
        wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=logger_step)

        batch_idx = 1
        total_loss = 0
        for i, (x, y) in enumerate(dataloaders["train"]):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()

            optimizer.step()
            batch_idx += 1
            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                # print("output: {}, target {}".format(output, y))
                # print(
                #     "total loss: {:.6f}, log interval {:2d}".format(
                #         total_loss, log_interval
                #     )
                # )
                cur_loss = total_loss / log_interval
                processed = min(
                    i * x.shape[0] + x.shape[0], len(dataloaders["train"].dataset)
                )
                print(
                    "Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch + 1,
                        processed,
                        len(dataloaders["train"].dataset),
                        100.0 * processed / len(dataloaders["train"].dataset),
                        cur_loss,
                    )
                )
                # --------------------------
                # log statistics of the loss
                wandb.log({"loss_train": cur_loss}, step=logger_step)
                logger_step += 1
                # Summary of best results
                if cur_loss < best_loss_train:
                    best_loss_train = cur_loss
                    wandb.run.summary["best_loss_train"] = best_loss_train
                # --------------------------
                total_loss = 0

        # Validation:
        model.eval()

        test_loss = 0
        total = 0
        for i, (x, y) in enumerate(dataloaders["validation"]):
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                output = model(x)
                test_loss += criterion(output, y).item() * x.size(0)
                total += x.size(0)

            # print("output: {}, target {}".format(output, y))
        test_loss = test_loss / total
        print("\nTest set: Average loss: {:.6f}\n".format(test_loss))
        # --------------------------
        # log statistics of the loss
        wandb.log({"loss_test": test_loss}, step=logger_step)
        # Summary of best results
        if test_loss < best_loss_test:
            best_loss_test = test_loss
            wandb.run.summary["best_loss_test"] = best_loss_test
            best_model_wts = copy.deepcopy(model.state_dict())
        # --------------------------

    # Report best results
    print("Best Val Acc: {:.4f}".format(best_loss_test))
    # Load best model weights
    model.load_state_dict(best_model_wts)
    # Return model and histories
    return model


def _test(model, test_loader, config):
    device = config.device
    model.eval()
    model.to(device)

    test_loss = 0
    total = 0
    for i, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            output = model(x)
            test_loss += F.mse_loss(output, y).item() * x.size(0)
            total += x.size(0)

    test_loss = test_loss / total
    print("\nTest set: Average loss: {:.6f}\n".format(test_loss))
    wandb.run.summary["best_test_accuracy"] = test_loss
