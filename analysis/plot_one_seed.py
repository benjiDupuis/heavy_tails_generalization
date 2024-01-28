import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from loguru import logger
from tqdm import tqdm

from last_point.utils import linear_regression
from last_point.simulation import asymptotic_constant
from last_point.utils import poly_alpha


def plot_bound(gen_tab, bound_tab, output_dir: str,
                sigma_values, alpha_values,
                    log_scale: bool = True, stem: str="",
                    xlabel:str="Accuracy error (%)"):
    
    output_dir = Path(output_dir)

    gen_tab = np.array(gen_tab)
    bound_tab = np.array(bound_tab)
    ids = np.where(gen_tab > 0)[0]
    gen_tab = gen_tab[ids]
    bound_tab = bound_tab[ids]
    sigma_values = np.array(sigma_values)[ids]
    alpha_values = np.array(alpha_values)[ids]

    a = max(min(gen_tab), min(bound_tab))
    b = min(max(gen_tab), max(bound_tab))

    # Colormap
    color_map = plt.cm.get_cmap('viridis_r')
    
    plt.figure()
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    output_path = (output_dir / ("estimated bound versus generalization_sigma_"  + stem )).with_suffix(".png")

    sc = plt.scatter(gen_tab,
                     bound_tab,
                     c=np.log(sigma_values) / np.log(10.),
                     cmap=color_map)
    cbar = plt.colorbar(sc)
    cbar.set_label("log(sigma)")

    logger.info(f"Saving a bound plot in {str(output_path)}")
    plt.savefig(str(output_path))
    plt.close()

    plt.figure()
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")

    output_path = (output_dir / ("estimated bound versus generalization_alpha_"  + stem )).with_suffix(".png")
    color_map = plt.cm.get_cmap('viridis_r')
    sc = plt.scatter(gen_tab,
                     bound_tab,
                     c=alpha_values,
                     cmap=color_map)
    cbar = plt.colorbar(sc)
    cbar.set_label(r"$\mathbf{\alpha}$")
    plt.xlabel(xlabel, weight="bold")
    logger.info(f"Saving a bound plot in {str(output_path)}")
    plt.savefig(str(output_path))
    plt.close()


def plot_one_seed(gen_grid, sigma_tab, alpha_tab, output_dir: str, deviation_grid = None):

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    n_sigma = len(sigma_tab)
    n_alpha = len(alpha_tab)

    logger.info(f"Saving all figures in {output_dir}")

    for s in tqdm(range(n_sigma)):

        plt.figure()
        if deviation_grid is not None:
            plt.errorbar(alpha_tab, gen_grid[s, :], yerr=deviation_grid[s, :])
        else:
            plt.scatter(alpha_tab, gen_grid[s, :])          

        # Saving the figure
        fig_name = (f"sigma_{sigma_tab[s]}").replace(".","_")
        output_path = (output_dir / fig_name).with_suffix(".png")
        plt.savefig(str(output_path))
        plt.close()

    for a in tqdm(range(n_alpha)):

            plt.figure()
            if deviation_grid is not None:
                plt.errorbar(sigma_tab, gen_grid[:, a], yerr=deviation_grid[:, a])
            else:
                plt.scatter(sigma_tab, gen_grid[:, a])
            plt.xscale("log")        

            # Saving the figure
            fig_name = (f"alpha_{alpha_tab[a]}").replace(".","_")
            output_path = (output_dir / fig_name).with_suffix(".png")
            plt.savefig(str(output_path))
            plt.close()
    
    # Finally: the linear regressions
    alpha_reg, correlation_reg = all_linear_regression(
        gen_grid,
        sigma_tab,
        alpha_tab
    )
    if all(alpha_reg[k] is not None for k in range(len(alpha_tab))):
        alpha_reg_path = (output_dir / "alpha_regression").with_suffix(".png")
        plt.figure()
        plt.plot(np.linspace(1,2, 100), np.linspace(1,2,100), color = "r")
        plt.scatter(alpha_tab, alpha_reg)
        plt.savefig(str(alpha_reg_path))
        plt.close()
    else:
        logger.warning("Linear regression did not work, probably due to negative generalization values")


def plot_gen_dim(json_path: str):
    """
    take the average result path and plot gen / dim
    """

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    num_exp = len(results.keys())
    logger.info(f"Found {num_exp} experiments")

    n_params_ids = {}
    i = 0
    for key in results.keys():
        if results[key]["n_params"] not in n_params_ids.keys():
            n_params_ids[results[key]["n_params"]] = i
            i += 1
    
    n_alpha = 1 + max(results[k]["id_alpha"] for k in results.keys())
    n_width = 1 + max(n_params_ids[k] for k in n_params_ids.keys())

    alpha_tab = np.zeros(n_alpha)
    n_params_tab = np.zeros(n_width)
    acc_gen_grid = np.zeros((n_width, n_alpha))
    acc_gen_grid_deviation = np.zeros((n_width, n_alpha))

    example_key = list(results.keys())[0]
    deviations: bool = ("acc_generalization_deviation" in list(results[example_key].keys()))

    for k in tqdm(results.keys()):

        n_params_tab[n_params_ids[results[k]["n_params"]]] = results[k]["n_params"]
        alpha_tab[results[k]["id_alpha"]] = results[k]["alpha"]

        acc_gen_grid[
            n_params_ids[results[k]["n_params"]],
            results[k]["id_alpha"]
        ] = results[k]["acc_generalization"]

        if deviations:
            acc_gen_grid_deviation[
                    n_params_ids[results[k]["n_params"]],
                    results[k]["id_alpha"]
                ] = results[k]["acc_generalization_deviation"]

    # Plot everything
    output_dir = json_path.parent / (json_path.parent.stem + "_figures")
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving figures in {str(output_dir)}")

    plot_one_seed(acc_gen_grid, n_params_tab, alpha_tab, str(output_dir), acc_gen_grid_deviation)



def analyze_one_seed(json_path: str):

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    num_exp = len(results.keys())
    logger.info(f"Found {num_exp} experiments")

    example_key = list(results.keys())[0]
    deviations: bool = ("acc_generalization_deviation" in list(results[example_key].keys()))

    # Collect n_alpha and n_sigma
    n_sigma = 1 + max(results[k]["id_sigma"] for k in results.keys())
    n_alpha = 1 + max(results[k]["id_alpha"] for k in results.keys())

    # Collect sigma_tab, alpha_tab, and the generalization grids
    sigma_tab = np.zeros(n_sigma)
    sigma_factor_tab = np.zeros(n_sigma)
    alpha_tab = np.zeros(n_alpha)
    normalization_tab = np.zeros(n_alpha)
    acc_gen_grid = np.zeros((n_sigma, n_alpha))
    gradient_grid = np.zeros((n_sigma, n_alpha))
    acc_gen_grid_deviation = np.zeros((n_sigma, n_alpha)) if deviations else None 

    gen_tab = []
    acc_bound_tab = []
    acc_tab = []
    sigma_values = []
    sigma_factor_values = []
    alpha_values = []
    
    for k in results.keys():

        # Estimate the actual value of the bound
        n = results[k]["n"]
        alpha = results[k]["alpha"]
        n_params = results[k]["n_params"]
        sigma = results[k]["sigma"]  # true value, without normalization by the dim
        sigma_factor = results[k]["sigma"] * np.sqrt(n_params)
        gradient = results[k]["gradient_mean"]
        gradient_unormalized = results[k]["gradient_mean_unormalized"]

        # TODO: this is ugly and suboptimal, find better
        sigma_tab[results[k]["id_sigma"]] = sigma
        sigma_factor_tab[results[k]["id_sigma"]] = sigma_factor
        alpha_tab[results[k]["id_alpha"]] = results[k]["alpha"]

        acc_gen_grid[
            results[k]["id_sigma"],
            results[k]["id_alpha"]
        ] = results[k]["acc_generalization"]

        gradient_grid[
            results[k]["id_sigma"],
            results[k]["id_alpha"]
        ] = results[k]["gradient_mean"]

        if deviations:
            acc_gen_grid_deviation[
                    results[k]["id_sigma"],
                    results[k]["id_alpha"]
                ] = results[k]["acc_generalization_deviation"]


        # Collect generalization error ad sigma and alpha, for colored plots
        gen_tab.append(results[k]["loss_generalization"])
        acc_tab.append(results[k]["acc_generalization"])
        sigma_values.append(sigma)
        sigma_factor_values.append(sigma_factor)
        alpha_values.append(results[k]["alpha"])

        # in pytorch sgd, decay is the true decay, not post lr
        decay = results[k]["decay"]
        horizon = results[k]["horizon"] + results[k]["n_ergodic"]
        lr = results[k]["eta"]

        # bs = results[k]["batch_size"]

        constant = asymptotic_constant(alpha, n_params)
        normalization_tab[results[k]["id_alpha"]] = constant

        # The factor tzo appearing here comes from an optimized PAC-Bayesian generalization bound for bounded losses
        acc_bound_tab.append(100. * np.sqrt((constant * horizon * lr * gradients)/\
                         (2. * n  * np.power(sigma, alpha))))

    # Plot everything
    output_dir = json_path.parent / (json_path.parent.stem + "_figures")
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving figures in {str(output_dir)}")

    plot_one_seed(acc_gen_grid, sigma_factor_tab, alpha_tab, str(output_dir), acc_gen_grid_deviation)
    plot_bound(acc_tab, acc_bound_tab, output_dir, sigma_factor_values,\
                alpha_values, log_scale=False, stem="accuracy")


def main(json_path: str, mode: str="all_plots"):

    if mode == "all_plots":
        analyze_one_seed(json_path)
    elif mode == "dim_regression":
        plot_alpha_dimension_regression(json_path)
    elif mode == "d_plots":
        plot_gen_dim(json_path)
    else:
        raise NotImplementedError()
 

if __name__ == "__main__":
     fire.Fire(main)










    

    