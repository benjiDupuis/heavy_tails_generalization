import json
import time
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

from data.dataset import get_full_batch_data, get_data_simple
from last_point.eval import eval
from last_point.model import fcnn, fcnn_num_params, NoisyCNN
from last_point.utils import robust_mean, poly_alpha


def run_one_simulation(horizon: int, 
                        d: int,
                        eta: float,
                        sigma: float,
                        alpha: float,
                        initialization: torch.Tensor,
                        data,
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
                        model: str = "cnn"):
    """
    Data format should be (x_train, y_train, x_val, y_val), all dataloaders
    stopping: whether or not stop the training at convergence,
    in that case the time horizon is still used to avoid infinite loops
    """

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # HACK TODO
    # This is a hack, the batch size experiment has to be properly implemented

    if batch_size < 0:
        logger.warning("No batch size")
    else:
        logger.warning(f"Batch size is {batch_size}")
    logger.info("Batch simulation script")

    # data
    train_loader = data[0]
    test_loader_eval = data[1]
    train_loader_eval = data[2]

    def cycle_loader(dataloader):
        while 1:
            for data in dataloader:
                yield data

    circ_train_loader = cycle_loader(train_loader)

    torch.manual_seed(seed)
    # np.random.seed(seed)
    if model == "fcnn":
        model = fcnn(d, width, depth, bias, n_classes)
    elif model == "cnn":
        model = NoisyCNN(width=width, out_features=n_classes)
    else:
        raise NotImplementedError(f"Model {model} not supported yet")
    logger.info(f"Used model: {model}")
    model.to(device)

    logger.info(f"On device {device}")
    
    opt = torch.optim.SGD(model.parameters(),
                           lr = eta,
                           weight_decay = decay)
    crit = nn.CrossEntropyLoss().to(device)
    crit_unreduced = nn.CrossEntropyLoss(reduction="none").to(device)

    loss_tab = []
    batch_loss_tab = []
    batch_acc_tab = []
    gen_tab = []
    accuracy_tab = []
    gradient_norm_list = []

    # Generate all noise
    # First reinitialize the seed
    # TODO: do something better than this hack and understand what is going on
    np.random.seed(int(str(time.time()).split(".")[1]))
    n_params = model.params_number()
    # noise = sigma * generate_levy_for_simulation(n_params, \
    #                                      horizon + n_ergodic,
    #                                      alpha,
    #                                      eta)
    # assert noise.shape == (horizon + n_ergodic, n_params),\
    #       (noise.shape, (horizon + n_ergodic, n_params))

    converged = False

    # Loop
    # for k in range(horizon + n_ergodic):
    for k, (x,y) in enumerate(circ_train_loader):

        if k >= horizon + n_ergodic:
            break

        if len(gen_tab) == n_ergodic:
            break

        x, y = x.to(device), y.to(device)

        # evaluation of the empirical loss
        # keep in mind that this is a full batch experiment
        opt.zero_grad()
        out = model(x)

        if k == 0:
            assert x.shape[0] == batch_size, (x.shape[0], batch_size)

        loss = crit(out, y)

        if torch.isnan(loss):
            logger.error('Loss has gone nan ❌')
            break

        # calculate the gradients
        loss.backward()

        # if accuracy_train >= 1.:
        #     converged = True

        if k % 1000 == 0:
            logger.info(f"Iteration number {k}, loss: {loss.item()}")
        batch_loss_tab.append(loss.item())
        batch_acc_tab.append(accuracy(out, y))

        # Validation if we are after the time horizon
        loss_val = None
        if k >= horizon or (converged and stopping):

            with torch.no_grad():
                te_hist, *_ = eval(test_loader_eval, model, crit_unreduced, opt)
                tr_hist, *_ = eval(train_loader_eval, model, crit_unreduced, opt)

            loss_tab.append((tr_hist[0], te_hist[0]))
            accuracy_tab.append((tr_hist[1], te_hist[1]))
            gen_tab.append(te_hist[0] - tr_hist[0])

        # Gradient step, put there to ensure initial acc are not corrupted
        opt.step()

        gradient_norm_list.append(model.gradient_l2_squared_norm())    

        # Adding the levy noise
        noise = sigma * generate_levy_for_simulation(n_params, \
                                         1,
                                         alpha,
                                         eta)
        with torch.no_grad():
            model.add_noise(torch.from_numpy(noise[0]).to(device))


    # Compute the estimated generalization at the end
    gen_tab = np.array(gen_tab)
    generalization = robust_mean(gen_tab)
    gradient_mean = float(robust_mean(np.array(gradient_norm_list))) if compute_gradients else "non_computed"
    gradient_mean_unormalized = float(np.array(gradient_norm_list).mean()) if compute_gradients else "non_computed"

    return float(generalization), loss_tab, accuracy_tab,\
           gradient_mean, converged, gradient_mean_unormalized, batch_acc_tab, n_params


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
                        normalization: bool = False,
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
                        batch_size: int = 32,
                        id_eta: int = 0,
                        model:str = "cnn"):
    """
    id_sigma and id_alpha are only there to be copied in the final JSON file.
    """

    logger.info("BATCH simulation")
    logger.info(f"Batch size: {batch_size}")
    assert batch_size >= 1, batch_size

    # HACK to avoid parsing issue of the classes
    if classes is not None:
        classes = [int(c) for c in classes]
    

    # Loading of the data
    if data_type == "mnist":
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        data = get_data_simple("mnist", "~/data", batch_size, 256, subset=subset, resize=resize, class_list=classes)

        # adapt the input dimension
        d = resize**2
        n_classes = 10 if classes is None else len(classes)

    elif data_type == "cifar10":
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        data = get_data_simple("cifar10", "~/data", batch_size, 512, subset=subset, resize=resize, class_list=classes)

        # adapt the input dimension
        d = resize**2 * 3
        n_classes = 10 if classes is None else len(classes)

    elif data_type == "fashion-mnist":
        np.random.seed(data_seed)
        torch.manual_seed(data_seed)
        data = get_data_simple("fashion-mnist", "~/data", batch_size, 256, subset=subset, resize=resize, class_list=classes)

        # adapt the input dimension
        d = resize**2
        n_classes = 10 if classes is None else len(classes)

    # TODO remove this HACK
    initialization = None 

    if model=="fcnn":
        n_params = fcnn_num_params(d, width, depth, bias, n_classes)
    elif model=="cnn":
        n_params = NoisyCNN(width=width, out_features=n_classes).params_number()
    else:
        raise NotImplementedError(f"Model {model} not supported")


    ######################################
    # We scale the value of sigma
    # wrt the sqrt of the number of parameters
    if scale_sigma:
        logger.warning("sigma has been renormalized by sqrt(n_params)")
        sigma = sigma / np.sqrt(n_params)
    ######################################

    # Normalization, if necessary
    normalization_factor = stable_normalization(alpha, n_params)
    if normalization:
        sigma_simu = normalization_factor * sigma   # Too dangerous
        logger.info(f"Normalization factor: {normalization_factor}")
    else:
        sigma_simu = sigma

    K_constant = asymptotic_constant(alpha, n_params)

    generalization, _, accuracy_tab,\
        gradient_mean, converged, \
          gradient_mean_unormalized, _, n_params= run_one_simulation(horizon, 
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
    accuracy_error = float(robust_mean(accuracy_error_tab_np))

    # Correct value of n
    n = len(data[0])
    n_val = len(data[2])

    logger.debug(f"K: {K_constant}, horizon: {horizon}, gradient_mean: {gradient_mean}, n: {n}, sigma: {sigma}, alpha: {alpha}")

    bound = np.sqrt(K_constant * horizon * gradient_mean / (n * np.power(sigma, alpha)))

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
        "estimated_bound": bound,
        "converged": converged,
        'final_train_accuracy': accuracy_tab[-1][0],
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
        result_dir.mkdir()

    result_path = (result_dir / f"result_{id_sigma}_{id_alpha}_{width}_{batch_size}_{id_eta}").with_suffix(".json")

    with open(str(result_path), "w") as result_file:
        json.dump(result_dict, result_file, indent = 2)
    logger.info(json.dumps(result_dict, indent=2))

    logger.info(f"Saved results JSON file in {str(result_path)}  ✅")


    
