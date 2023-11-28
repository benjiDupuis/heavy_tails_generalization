import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from typing import Tuple

from last_point.model import fcnn
from last_point.utils import accuracy
from levy.levy import generate_levy_for_simulation


def run_one_simulation(horizon: int, 
                        d: int,
                        eta: float,
                        sigma: float,
                        alpha: float,
                        initialization: torch.Tensor,
                        data: Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor],
                        n_ergodic: int = 100,
                        n_classes: int = 2,
                        momentum: float = 0.,
                        depth: int = 0,
                        width: int = 50):
    """
    Data format should be (x_train, y_train, x_val, y_val)
    """

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Seed
    # TODO: does this affect the generation of the levy processes?
    # TODO: if it does, can we still trust the experimental results?
    # TODO: how does the levy process generation use the seed?
    torch.manual_seed(42)

    # Define data
    assert len(data) == 4, len(data)
    x_train = data[0].to(device)
    y_train = data[1].to(device)
    x_val = data[2].to(device)
    y_val = data[3].to(device)
    assert x_train.ndim == 2
    assert y_train.ndim == 1
    assert x_val.ndim == 2
    assert y_val.ndim == 1
    assert x_train.shape[0] == y_train.shape[0]
    assert x_train.shape[1] == x_val.shape[1]

    n = x_train.shape[0]

    # Define model, loss and optimizer
    # if depth == 0:
    #     model = LinearModel(d, n_classes = n_classes).to(device) 
    #     with torch.no_grad():
    #         model.initialization(initialization)
    # else:
    #     model = fcnn(depth=depth, width=width)
    model = fcnn(d, width, depth, False, n_classes)
    
    opt = torch.optim.SGD(model.parameters(),
                           lr = eta, momentum=momentum)
    crit = nn.CrossEntropyLoss().to(device)

    loss_tab = []
    gen_tab = []
    accuracy_tab = []

    # Generate all noise
    n_params = model.params_number()
    noise = sigma * generate_levy_for_simulation(n_params, \
                                         horizon + n_ergodic,
                                         alpha,
                                         eta)
    assert noise.shape == (horizon + n_ergodic, n_params),\
          (noise.shape, (horizon + n_ergodic, n_params))

    # Loop
    for k in range(horizon + n_ergodic):

        # evaluation of the empirical loss
        # keep in mind that this is a full batch experiment
        opt.zero_grad()
        out = model(x_train)
        # logger.info(f"out: {out}")
        # logger.info(f"means: {initialization}")

        assert out.shape == (n, n_classes), out.shape
        loss = crit(out, y_train)

        if torch.isnan(loss):
            logger.error('Loss has gone nan ❌')
            break

        # calculate the gradients
        loss.backward()

        # Gradient step
        opt.step()

        # Validation if we are after the time horizon
        loss_val = None
        if k >= horizon:
            with torch.no_grad():
                out_val = model(x_val)
                loss_val = crit(out_val, y_val).item()

        with torch.no_grad():
            # Logging
            loss_tab.append((loss.item(), loss_val))
            accuracy_train = accuracy(out, y_train)
            accuracy_val = None

            if k >= horizon:
                gen_tab.append(loss_val - loss.item())
                accuracy_val = accuracy(out_val, y_val)

            accuracy_tab.append((accuracy_train, accuracy_val))

        # Adding the levy noise
        with torch.no_grad():
            model.add_noise(torch.from_numpy(noise[k]).to(device))

        if k == 0:
            logger.debug(f"Initial train accuracy: {accuracy_train}")
            logger.debug(f"Initial test accuracy: {accuracy_train}")

    # Compute the estimated generalization at the end
    gen_tab = np.array(gen_tab)
    generalization = gen_tab.mean()

    return generalization, loss_tab, accuracy_tab,\
          (out.detach().cpu().numpy(), out_val.detach().cpu().numpy())



