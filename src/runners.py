from typing import List, Tuple, cast

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import trange

Dataset = List[Tuple[torch.Tensor, torch.Tensor]]


def train_epoch(
    data_loader: Dataset,
    model: nn.Module,
    optimiser: torch.optim.optimizer.Optimizer,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[nn.Module, float, float, float]:
    # set model to training mode. This is important because some layers behave differently during
    # training and testing
    model = model.train(True)
    model = model.to(device)  # (if not already do)

    # stats
    loss_total = 0.0
    f1_total = 0.0
    oa_total = 0.0

    # iterate over dataset
    p_bar = trange(len(data_loader))
    for idx, (data, target) in enumerate(data_loader):
        # put data and target onto correct device (if not already do)
        data, target = data.to(device), target.to(device)

        # reset gradients
        optimiser.zero_grad()

        # forward pass
        pred = model(data)

        # loss
        loss = criterion(pred, target)

        # backward pass
        loss.backward()

        # parameter update
        optimiser.step()

        # stats update
        loss_total += loss.item()
        f1_total += f1_score(
            target.detach().cpu().numpy().flatten(),
            pred.argmax(1).detach().cpu().numpy().flatten(),
            zero_division=0,
        )
        oa_total += torch.mean(cast(torch.Tensor, pred.argmax(1) == target).float()).item()

        # format progress bar
        p_bar.set_description(
            "Loss: {:.2f}, F1: {:.2f}, OA: {:.2f}".format(
                loss_total / (idx + 1), 100 * f1_total / (idx + 1), 100 * oa_total / (idx + 1)
            )
        )
        p_bar.update(1)

    p_bar.close()

    # normalise stats
    loss_total /= len(data_loader)
    f1_total /= len(data_loader)
    oa_total /= len(data_loader)

    return model, loss_total, f1_total, oa_total


def validate_epoch(
    data_loader: Dataset, model: nn.Module, device: torch.device, criterion: nn.Module
) -> Tuple[float, float, float]:
    # set model to evaluation mode
    model = model.train(False)
    model = model.to(device)  # (if not already do)

    # stats
    loss_total = 0.0
    f1_total = 0.0
    oa_total = 0.0

    # iterate over dataset
    p_bar = trange(len(data_loader))
    for idx, (data, target) in enumerate(data_loader):
        with torch.no_grad():
            # put data and target onto correct device (if not already do)
            data, target = data.to(device), target.to(device)

            # forward pass
            pred = model(data)

            # loss
            loss = criterion(pred, target)

            # stats update
            loss_total += loss.item()
            f1_total += f1_score(
                target.detach().cpu().numpy().flatten(),
                pred.argmax(1).detach().cpu().numpy().flatten(),
                zero_division=0,
            )
            oa_total += torch.mean(cast(torch.Tensor, pred.argmax(1) == target).float()).item()

            # format progress bar
            p_bar.set_description(
                "Loss: {:.2f}, F1: {:.2f}, OA: {:.2f}".format(
                    loss_total / (idx + 1), 100 * f1_total / (idx + 1), 100 * oa_total / (idx + 1)
                )
            )
            p_bar.update(1)

    p_bar.close()

    # normalise stats
    loss_total /= len(data_loader)
    f1_total /= len(data_loader)
    oa_total /= len(data_loader)

    return loss_total, f1_total, oa_total
