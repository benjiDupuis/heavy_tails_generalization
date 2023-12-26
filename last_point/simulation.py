import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy.special import gamma
from typing import Tuple

from last_point.model import fcnn
from last_point.utils import accuracy
from levy.levy import generate_levy_for_simulation

from data.dataset import get_full_batch_data
from last_point.gaussian_mixture import sample_standard_gaussian_mixture
from last_point.model import fcnn, SinusFCNN


def run_one_simulation(horizon: int, 
                        d: int,
                        eta: float,
                        sigma: float,
                        alpha: float,
                        initialization: torch.Tensor,
                        data: Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor],
                        n_ergodic: int = 100,
                        n_classes: int = 2,
                        decay: float = 0.,
                        depth: int = 0,
                        width: int = 50,
                        seed: int = 42,
                        compute_gradients: bool = False,
                        bias: bool = False):
    """
    Data format should be (x_train, y_train, x_val, y_val)
    """

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define data
    assert len(data) == 4, len(data)
    x_train = data[0].to(device)
    y_train = data[1].to(device)
    x_val = data[2].to(device)
    y_val = data[3].to(device)
    #assert x_train.ndim == 2
    #assert y_train.ndim == 1
    #assert x_val.ndim == 2
    #assert y_val.ndim == 1
    assert x_train.shape[0] == y_train.shape[0]
    assert x_train.shape[1] == x_val.shape[1]

    n = x_train.shape[0]

    # Seed
    # TODO: does this affect the generation of the levy processes?
    # TODO: if it does, can we still trust the experimental results?
    # TODO: how does the levy process generation use the seed?
    torch.manual_seed(seed)
    # np.random.seed(seed)
    model = fcnn(d, width, depth, bias, n_classes)
    # model = SinusFCNN(depth=depth, width=width, input_dim=d, bias=True)
    
    opt = torch.optim.SGD(model.parameters(),
                           lr = eta,
                           weight_decay = decay)
    crit = nn.CrossEntropyLoss().to(device)

    loss_tab = []
    gen_tab = []
    accuracy_tab = []
    gradient_norm_list = []

    # Generate all noise
    # First reinitialize the seed
    # TODO: do something better than this hack and understand what is going on
    np.random.seed(np.random.randint(10000))
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

        if k == 0:
            logger.debug(f"Initial train accuracy: {accuracy_train}")
            logger.debug(f"Initial test accuracy: {accuracy_train}")

        # Gradient step, put there to ensure initial acc are not corrupted
        opt.step()

        if compute_gradients and k >= horizon:
            with torch.no_grad():
                gradient_norm_list.append(model.gradient_l2_squared_norm())

        # Adding the levy noise
        with torch.no_grad():
            model.add_noise(torch.from_numpy(noise[k]).to(device))

    # Compute the estimated generalization at the end
    gen_tab = np.array(gen_tab)
    generalization = gen_tab.mean()
    gradient_mean = float(np.array(gradient_norm_list).mean()) if compute_gradients else "non_computed"

    return float(generalization), loss_tab, accuracy_tab,\
          None, gradient_mean


def stable_normalization(alpha: float, d: float) -> float:

        assert alpha >= 1., alpha
        assert alpha <= 2., alpha

        if alpha == 2.:
            # asymptotic development of gamma(1-s)
            # using EUler reflection formula
            # TODO: recheck this
            alpha_factor = 1.
        else:
            alpha_factor = np.power((2. - alpha) * gamma(1. - alpha/2.) / (alpha), 1./alpha)

        dimension_factor = np.power(d, 0.5 - 1./alpha)
        norm_factor = alpha_factor / (np.sqrt(2.) * dimension_factor)

        return norm_factor

def asymptotic_constant(alpha: float, d: float) -> float:

    if alpha == 2.:
        # asymptotic development of gamma(1-s)
        # using EUler reflection formula
        # TODO: recheck this
        num_alpha = 2.
    else:
        num_alpha = (2. - alpha) * gamma(1. - alpha / 2.)

    den_alpha = alpha * np.power(2., alpha / 2.)
    dimension_factor = np.power(d, 1. - alpha / 2.)

    return num_alpha * dimension_factor / den_alpha


def run_and_save_one_simulation(result_dir: str,
                        horizon: int, 
                        d: int,
                        eta: float,
                        sigma: float,
                        alpha: float,
                        n: int = 1000,
                        n_val: int = 1000,
                        n_ergodic: int = 100,
                        n_classes: int = 2,
                        decay: float = 0.,
                        depth: int = 0,
                        width: int = 50,
                        data_seed: int = 1,
                        model_seed: int = 42,
                        normalization: bool = True,
                        id_sigma: int = 0,
                        id_alpha: int = 0,
                        compute_gradient: bool = False,
                        bias: bool = False,
                        data_type: str = "mnist",
                        subset: float = 0.01,
                        resize: int = 14):
    """
    id_sigma and id_alpha are only there to be copied in the final JSON file.
    """
    
    # Generate the data
    # First the seed is set, so that each training will have the same data
    if data_type == "gaussian":
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        n_per_class_train = n // n_classes
        x_train, y_train, means = sample_standard_gaussian_mixture(d, n_per_class_train)
        n_per_class_val = n_val // n_classes
        x_val, y_val, _ = sample_standard_gaussian_mixture(d, n_per_class_val, 
                                                            random_centers=False, 
                                                            means_deterministic=means)

        data = (x_train, y_train, x_val, y_val)
        print(data)

    elif data_type == "mnist":
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        data = get_full_batch_data("mnist", "~/data", subset_percentage=subset, resize=resize)

        # adapt the input dimension
        d = resize**2
        n_classes = 10

    # TODO remove this hack
    initialization = None 

    # TODO: remove this hack
    model_temp = fcnn(d, width, depth, bias, n_classes)
    n_params = model_temp.params_number()

    # Normalization, if necessary
    normalization_factor = stable_normalization(alpha, n_params)
    if normalization:
        sigma_simu = normalization_factor * sigma
        logger.info(f"Normalization factor: {normalization_factor}")
    else:
        sigma_simu = sigma

    K_constant = asymptotic_constant(alpha, n_params)

    generalization, _, accuracy_tab,\
          _, gradient_mean = run_one_simulation(horizon, 
                                    d,
                                    eta,
                                    sigma_simu,
                                    alpha,
                                    initialization,
                                    data,
                                    n_ergodic,
                                    n_classes,
                                    decay,
                                    depth,
                                    width,
                                    seed=model_seed,
                                    compute_gradients=compute_gradient,
                                    bias=bias)

    # Estimating accuracy error over last iterations
    accuracy_error_tab = accuracy_tab[-n_ergodic:]
    assert len(accuracy_error_tab) == n_ergodic
    assert all(accuracy_error_tab[k][0] is not None for k in range(n_ergodic))
    assert all(accuracy_error_tab[k][1] is not None for k in range(n_ergodic))

    accuracy_error_tab_np = np.array([
        accuracy_error_tab[k][0] - accuracy_error_tab[k][1] for k in range(n_ergodic)
    ])
    accuracy_error = float(100. * accuracy_error_tab_np.mean())

    bound = np.sqrt(K_constant * gradient_mean / (n * decay * np.power(sigma, alpha)))

    result_dict = {
        "horizon": horizon, 
        "input_dimension": d,
        "n": n,
        "n_val": n_val,
        "data_seed": data_seed,
        "model_seed": model_seed,
        "eta": eta,
        "sigma": sigma,
        "alpha": alpha,
        "n_ergodic": n_ergodic,
        "n_classes": n_classes,
        "decay": decay,
        "depth": depth,
        "width": width,
        "loss_generalization": float(generalization),
        "acc_generalization": accuracy_error,
        "id_sigma": id_sigma,
        "id_alpha": id_alpha,
        "normalization_factor": normalization_factor,
        "normalization": int(normalization),
        "gradient_mean": gradient_mean,
        "K_constant": K_constant,
        "n_params": n_params,
        "bias": bias,
        "estimated_bound": bound
    }

    result_dir = Path(result_dir)
    if not result_dir.is_dir():
        result_dir.mkdir()

    result_path = (result_dir / f"result_{id_sigma}_{id_alpha}").with_suffix(".json")

    with open(str(result_path), "w") as result_file:
        json.dump(result_dict, result_file, indent = 2)

    
