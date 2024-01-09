import datetime
import json
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy.special import gamma
from typing import Tuple

from data.dataset import get_full_batch_data
from last_point.gaussian_mixture import sample_standard_gaussian_mixture
from last_point.model import fcnn, fcnn_num_params
from last_point.simulation import asymptotic_constant, run_one_simulation

CONVERGENCE_SEED = int(str(time.time()).split(".")[1])

def plot_loss_alpha(result_dir: str):
    pass

def main(result_dir: str='tests_directory',
          horizon: int=0, 
          d: int=10,
          eta: float=0.01,
          sigma: float=20.,
          alpha: float=1.8,
          n: int = 1000,
          n_val: int = 1000,
          n_ergodic: int = 10000,
          n_classes: int = 2,
          decay: float = 0.001,
          depth: int = 1,
          width: int = 100,
          data_seed: int = CONVERGENCE_SEED,
          model_seed: int = CONVERGENCE_SEED + 1,
          normalization: bool = False,
          id_sigma: int = 0,
          id_alpha: int = 0,
          compute_gradient: bool = False,
          bias: bool = False,
          data_type: str = "gaussian",
          subset: float = 0.01,
          resize: int = 28,
          classes: list = None,
          stopping: bool = False):
    """
    id_sigma and id_alpha are only there to be copied in the final JSON file.
    """

    # HACK to avoid parsing issue of the classes
    if classes is not None:
        classes = [int(c) for c in classes]
    
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
        data = get_full_batch_data("mnist", "~/data", subset_percentage=subset, resize=resize, class_list=classes)

        # adapt the input dimension
        d = resize**2
        n_classes = 10

    # TODO remove this hack
    initialization = None 

    n_params = fcnn_num_params(d, width, depth, bias, n_classes)

    ######################################
    # We scale the value of sigma
    # wrt the sqrt of the number of parameters
    sigma = sigma / np.sqrt(n_params)
    ######################################

    K_constant = asymptotic_constant(alpha, n_params)

    generalization, loss_tab, accuracy_tab,\
          _, gradient_mean, converged = run_one_simulation(horizon, 
                                    d,
                                    eta,
                                    sigma,
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
                                    stopping=stopping)
    
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
    accuracy_error = float(100. * accuracy_error_tab_np.mean())

    if compute_gradient:
      bound = np.sqrt(K_constant * gradient_mean / (n * decay * np.power(sigma, alpha)))
    else:
      bound = None

    # Correct value of n
    n = data[0].shape[0]
    n_val = data[2].shape[0]

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
        "normalization_factor": 'not computed',
        "normalization": int(normalization),
        "gradient_mean": gradient_mean,
        "K_constant": K_constant,
        "n_params": n_params,
        "bias": bias,
        "estimated_bound": bound,
        "converged": converged,
        'final_train_accuracy': accuracy_tab[-1][0].item(),
        'class_list': classes
    }

    result_dir = Path(result_dir) / str(datetime.datetime.now()).replace(" ", "_").replace(":", "_").split(".")[0]
    if not result_dir.is_dir():
        result_dir.mkdir(parents=True, exist_ok=True)

    result_path = (result_dir / f"result_{id_sigma}_{id_alpha}").with_suffix(".json")

    with open(str(result_path), "w") as result_file:
        json.dump(result_dict, result_file, indent = 2)

    # plots
    iterations = len(accuracy_tab)

    train_accs = [accuracy_tab[i][0] for i in range(iterations)]
    val_accs = [accuracy_tab[i][1] for i in range(iterations)]

    train_losses = [loss_tab[i][0] for i in range(iterations)]
    val_losses = [loss_tab[i][1] for i in range(iterations)]

    # evolution  of the training and validation losses
    plt.figure()
    plt.plot(np.arange(iterations), train_losses, label="Train BCE")
    plt.plot(np.arange(iterations), val_losses, label="Validation BCE")
    fig_name = (f"loss_sigma_{sigma}_alpha_{alpha}").replace(".","_")
    output_path = (result_dir / fig_name).with_suffix(".png")
    plt.legend()
    logger.info(f'Saving loss plot in {str(output_path)}')
    plt.savefig(str(output_path))
    plt.close()      

    # evolution of both accuracies
    plt.figure()
    plt.plot(np.arange(iterations), train_accs, label="Train accuracy")
    plt.plot(np.arange(iterations), val_accs, label="Test accuracy")
    fig_name = (f"accuracies_sigma_{sigma}_alpha_{alpha}").replace(".","_")
    output_path = (result_dir / fig_name).with_suffix(".png")
    plt.legend()
    logger.info(f'Saving accuracy plot in {str(output_path)}')
    plt.savefig(str(output_path))
    plt.close()

    return train_accs, val_accs

if __name__ == "__main__":
    fire.Fire(main)