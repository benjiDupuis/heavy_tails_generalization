import json
import time
from pathlib import Path

import fire
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
from last_point.model import fcnn, fcnn_num_params
from last_point.utils import robust_mean, poly_alpha


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
                        bias: bool = False,
                        stopping: bool = False,
                        batch_size: int = -1,
                        model: str =  "fcnn"):
    """
    This is the main simulation script, the arguments are:
    horizon: number of iterations
    d: input_dimension, if needed
    eta: learning random_centers
    sigma: noise scale
    alpha: tail-index to be used in the simulation
    initialization: potential initialization of the model, not used in the current version of our work
    data: Data format should be (x_train, y_train, x_val, y_val)
    n_ergodic: how many iterations, at the end of training, are used to compute the average generalizatio error
    n_classes: number of classes, if needed
    decay: l2 regularization coefficient
    depth: depth of the network
    width: width of the network
    seed: random seed
    compute_gradients: whether or not compute the squared gradient gradient_norm_list
    bias: whether or not use bais in the networks
    stopping: whether or not stop the training at convergence
    batch_size: batch size of training, -1 corresponds to full batch training
    """

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if batch_size < 0:
        logger.warning("No batch size")
    else:
        logger.warning(f"Batch size is {batch_size}")

    # Define data
    assert len(data) == 4, len(data)
    x_train = data[0].to(device)
    y_train = data[1].to(device)
    x_val = data[2].to(device)
    y_val = data[3].to(device)
    assert x_train.shape[0] == y_train.shape[0]
    assert x_train.shape[1] == x_val.shape[1]

    n = x_train.shape[0]

    torch.manual_seed(seed)
    if model == "fcnn":
        model = fcnn(d, width, depth, bias, n_classes)
        model.to(device)
    else:
        raise NotImplementedError(f"model {model} not taken into account yet")

    logger.info(f"On device {device}")
    
    opt = torch.optim.SGD(model.parameters(),
                           lr = eta,
                           weight_decay = decay)
    crit = nn.CrossEntropyLoss().to(device)

    loss_tab = []
    gen_tab = []
    accuracy_tab = []
    gradient_norm_list = []

    # Generate all noise
    # First reinitialize the seed, to avoid introducing additional bias in the simulations
    np.random.seed(int(str(time.time()).split(".")[1]))
    n_params = model.params_number()
    noise = sigma * generate_levy_for_simulation(n_params, \
                                         horizon + n_ergodic,
                                         alpha,
                                         eta)
    assert noise.shape == (horizon + n_ergodic, n_params),\
          (noise.shape, (horizon + n_ergodic, n_params))

    converged = False

    logger.info(f"Shape of training tensor: {x_train.shape}")

    # Loop
    for k in range(horizon + n_ergodic):

        # evaluation of the empirical loss
        # keep in mind that this is a full batch experiment
        opt.zero_grad()

        if batch_size >= 1:
            batch = np.random.choice(np.arange(x_train.shape[0]), batch_size, replace=False)
        else:
            batch = np.arange(x_train.shape[0])

        out = model(x_train[batch,...])

        if k == 0:
            logger.info(f"Shape of the output of the model: {out.shape}")
            logger.info(f"Batch_size: {batch_size}")

        # assert out.shape == (n, n_classes), out.shape
        loss = crit(out, y_train[batch,...])

        if torch.isnan(loss):
            logger.error('Loss has gone nan ❌')
            break

        # calculate the gradients
        loss.backward()

        # Logging
        with torch.no_grad():
            if batch_size >= 0:
                out_acc = model(x_train)
                accuracy_train = accuracy(out_acc, y_train)
            else:
                accuracy_train = accuracy(out, y_train)

        if accuracy_train >= 1.:
            converged = True

        # Validation if we are after the time horizon
        loss_val = None
        if k >= horizon or (converged and stopping):
            with torch.no_grad():
                out_val = model(x_val)
                loss_val = crit(out_val, y_val).item()
                loss_tab.append((loss.item(), loss_val))
                accuracy_val = None

        with torch.no_grad():
            if k >= horizon or (converged and stopping):
                gen_tab.append(loss_val - loss.item())
                accuracy_val = accuracy(out_val, y_val)
                accuracy_tab.append((accuracy_train, accuracy_val))

        if k == 0:
            logger.debug(f"Initial train accuracy: {accuracy_train}")

        # Gradient step, put there to ensure initial acc are not corrupted
        opt.step()

        with torch.no_grad():
            gradient_norm_list.append(model.gradient_l2_squared_norm())
        

        # Adding the levy noise
        with torch.no_grad():
            model.add_noise(torch.from_numpy(noise[k]).to(device))

        if len(gen_tab) == n_ergodic:
            break

    # Compute the estimated generalization at the end
    gen_tab = np.array(gen_tab)
    if torch.isnan(loss):
        generalization = gen_tab.mean()
        gradient_mean = gradient_mean.mean() if compute_gradients else "non_computed"
    else:
        generalization = robust_mean(gen_tab)
        gradient_mean = float(robust_mean(np.array(gradient_norm_list))) if compute_gradients else "non_computed"
    gradient_mean_unormalized = float(np.array(gradient_norm_list).mean()) if compute_gradients else "non_computed"

    return float(generalization), loss_tab, accuracy_tab,\
          None, gradient_mean, converged, gradient_mean_unormalized


def asymptotic_constant(alpha: float, d: float) -> float:
    '''
    compute the constqnt K_{\alpha, d} appearing in the bounds
    '''

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
                        d: int=2,
                        eta: float=0.01,
                        sigma: float=1.,
                        alpha: float=1.8,
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
                        compute_gradient: bool = True,
                        bias: bool = False,
                        data_type: str = "mnist",
                        subset: float = 0.01,
                        resize: int = 28,
                        classes: list = None,
                        stopping: bool = False,
                        scale_sigma: bool = True,
                        batch_size: int = -1,
                        id_eta: int = 0,
                        model: str = "fcnn"):

    """
    result_dir: where to save the results
    horizon: number of iterations
    d: input_dimension, unused in the current version of the work, as it is computed from the dataset
    eta: learning random_centers
    sigma: noise scale
    alpha: tail index to be used in the simulations
    n: unused in the current version of the work
    n_val: unused in the current version of the work
    n_ergodic: how many iterations, at the end of training, are used to compute the average generalizatio error
    n_classes: number of classes, unused in the current version of the work
    decay: l2 regularization coefficient
    depth: depth of the network
    width: width of the network
    data_seed: random seed for data
    model_seed: random seed for model: 
    id_sigma: identifier of the noise scale being used
    id_alpha: identifier of the tail index being used
    compute_gradients: whether or not compute the squared gradient gradient_norm_list
    bias: whether or not use bais in the networks
    data_type: which dataset to use, can be "mnist" or "fashion-mnist", dataset script provides support for more datasets
    subset: proportion of the chosen data to use (1=100%)
    resize: optional resize of the images, if applicable, not used in our experiments
    classes: potential selection of the classes to be used in the experiment, all classes are used by default
    stopping: whether or not stop the training at convergence
    sclae_sigma: whether or not to scale the noise scale by sqrt(n_params), according to our theory.
    batch_size: batch size of training, -1 corresponds to full batch training
    id_eta: identifier of the learning rate being used
    """
    
    if classes is not None:
        classes = [int(c) for c in classes]


    if data_type == "mnist":
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        data = get_full_batch_data("mnist", "~/data", subset_percentage=subset, resize=resize, class_list=classes)

        # adapt the input dimension
        d = resize**2
        n_classes = 10 if classes is None else len(classes)

    elif data_type == "fashion-mnist":
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        data = get_full_batch_data("fashion-mnist", "~/data", subset_percentage=subset, resize=resize, class_list=classes)

        # adapt the input dimension
        d = resize**2
        n_classes = 10 if classes is None else len(classes)

    elif data_type == "cifar10":
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        data = get_full_batch_data("cifar10", "~/data", subset_percentage=subset, resize=resize, class_list=classes)

        # adapt the input dimension
        d = 3 * resize**2
        n_classes = 10 if classes is None else len(classes)

    else: 
        raise NotImplementedError(f"Model {model} not supported yet")

    initialization = None 

    n_params = fcnn_num_params(d, width, depth, bias, n_classes)

    ######################################
    # We scale the value of sigma
    # wrt the sqrt of the number of parameters
    if scale_sigma:
        logger.warning("sigma has been renormalized by sqrt(n_params)")
        sigma = sigma / np.sqrt(n_params)
    ######################################

    # Normalization, if necessary
    sigma_simu = sigma

    K_constant = asymptotic_constant(alpha, n_params)

    generalization, _, accuracy_tab,\
          _, gradient_mean, converged, \
          gradient_mean_unormalized = run_one_simulation(horizon, 
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
                                    bias=bias,
                                    stopping=stopping,
                                    batch_size=batch_size,
                                    model=model)
    
    if converged:
            logger.info('Experiment converged!')


    # Estimating accuracy error over last iterations
    accuracy_error_tab = accuracy_tab[-n_ergodic:]
    assert len(accuracy_error_tab) == n_ergodic
    assert all(accuracy_error_tab[k][0] is not None for k in range(n_ergodic))
    assert all(accuracy_error_tab[k][1] is not None for k in range(n_ergodic))

    accuracy_error_tab_np = np.array([
        accuracy_error_tab[k][0] - accuracy_error_tab[k][1] for k in range(n_ergodic)
    ])
    accuracy_error = float(100. * robust_mean(accuracy_error_tab_np))

    # Correct value of n
    n = data[0].shape[0]
    n_val = data[2].shape[0]

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
        "normalization": int(normalization),
        "gradient_mean": gradient_mean,
        "K_constant": K_constant,
        "n_params": n_params,
        "bias": bias,
        "estimated_bound": bound,
        "converged": converged,
        'final_train_accuracy': accuracy_tab[-1][0].item(),
        "acc_gen_normalized": accuracy_error / np.sqrt(poly_alpha(alpha)),
        "gradient_mean_unormalized": gradient_mean_unormalized,
        "resize": resize,
        "scale_sigma": scale_sigma,
        "normalization": normalization,
        "stopping": stopping,
        "batch_size": batch_size,
        "id_eta": id_eta
    }

    result_dir = Path(result_dir)
    if not result_dir.is_dir():
        result_dir.mkdir(exist_ok=True, parents=True)

    eta_name = str(eta).replace(".","_")
    rname = f"result_{id_sigma}_{id_alpha}_{width}_{eta_name}_{batch_size}"
    result_path = (result_dir / rname).with_suffix(".json")

    logger.info(f"Saving results JSON file in {str(result_path)}")

    with open(str(result_path), "w") as result_file:
        json.dump(result_dict, result_file, indent = 2)



if __name__ == "__main__":
    fire.Fire(run_and_save_one_simulation) 
