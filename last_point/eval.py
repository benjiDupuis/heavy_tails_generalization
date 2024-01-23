import time

import numpy as np
import torch
import torch.nn as nn
from loguru import logger



def accuracy(out, y):
    _, pred = out.max(1)
    correct = pred.eq(y)
    return 100 * correct.sum().cpu().float() / y.size(0)

@torch.no_grad()
def recover_eval_tensors(dataloader):

    final_x, final_y = [], []

    for x, y in dataloader:

        final_x.append(x)
        final_y.append(y)

    return torch.cat(final_x, 0), torch.cat(final_y, 0)


@torch.no_grad()
def eval_on_tensors(x, y, net, criterion):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    x, y = x.to(device), y.to(device)

    out = net(x)

    # probabilities = torch.softmax(out, dim=1).numpy()
    # surrogate_features = (-1.) * probabilities[np.arange(y.shape[0]), y.numpy()]


    losses = criterion(out, y)
    prec = accuracy(out, y)

    hist = [
        losses.sum().cpu().item() / x.shape[0],
        prec.item()
    ]

    return hist, losses, out


@torch.no_grad()
def eval(eval_loader, net, criterion, opt, eval: bool = False):
    """
    WARNING: criterion is not used anymore
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net.eval()

    # run over both test and train set
    total_size = 0
    total_loss = 0
    total_acc = 0
    losses = []
    outputs = []

    for x, y in eval_loader:

        # loop over dataset
        x, y = x.to(device), y.to(device)

        out = net(x)

        losses_unreduced = criterion(out, y)
        prec = accuracy(out, y)
        bs = x.size(0)

        total_size += int(bs)
        total_loss += float(losses_unreduced.sum().cpu().item()) 
        total_acc += float(prec) * bs

        losses.append(losses_unreduced)
        outputs.append(out.flatten())

    hist = [
        total_loss / total_size,
        total_acc / total_size,
    ]

    # losses: list of tensors of shape (batch_size)
    # We concatenate it into a tensor of shape (len(eval_loader))
    losses = torch.cat(losses)
    outputs = torch.cat(outputs)

    return hist, losses, outputs
