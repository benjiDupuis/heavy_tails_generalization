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

def all_linear_regression(
                        gen_grid: np.ndarray,
                        sigma_tab: np.ndarray,
                        alpha_tab: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Returns the regression of the gen with respect to log(1/sigma), for each alpha
        and the regression of the gen with respect to alpha, for each sigma
        """
        n_alpha = len(alpha_tab)
        n_sigma = len(sigma_tab)

        # Regression gen/log(1/sigma)
        alpha_reg = np.zeros(n_alpha)
        for a in range(n_alpha):
            alpha_reg[a] = linear_regression(np.log(1./sigma_tab), gen_grid[:, a])

        # Regression gen/alpha
        correlation_reg = np.zeros(n_sigma)
        for s in range(n_sigma):
            correlation_reg[s] = linear_regression(alpha_tab, gen_grid[s, :])

        return alpha_reg, correlation_reg

def plot_bound(gen_tab, bound_tab, output_dir: str,
                sigma_values, alpha_values,
                    log_scale: bool = True, stem: str=""):
    
    output_dir = Path(output_dir)

    # Colormap
    color_map = plt.cm.get_cmap('RdYlBu')
    
    plt.figure()
    output_path = (output_dir / ("estimated bound versus generalization_sigma_"  + stem )).with_suffix(".png")
    # plt.scatter(gen_tab, bound_tab)
    sc = plt.scatter(gen_tab,
                     bound_tab,
                     c=np.log(sigma_values) / np.log(10.),
                     cmap=color_map)
    cbar = plt.colorbar(sc)
    cbar.set_label("log(sigma)")
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.title("estimated bound versus generalization")
    logger.info(f"Saving a bound plot in {str(output_path)}")
    plt.savefig(str(output_path))
    plt.close()

    plt.figure()
    output_path = (output_dir / ("estimated bound versus generalization_alpha_"  + stem )).with_suffix(".png")
    # plt.scatter(gen_tab, bound_tab)
    sc = plt.scatter(gen_tab,
                     bound_tab,
                     c=alpha_values,
                     cmap=color_map)
    cbar = plt.colorbar(sc)
    cbar.set_label("alpha")
    if log_scale:
        plt.yscale("log")
        plt.xscale("log")
    plt.title("estimated bound versus generalization")
    logger.info(f"Saving a bound plot in {str(output_path)}")
    plt.savefig(str(output_path))
    plt.close()


def plot_one_seed(gen_grid, sigma_tab, alpha_tab, output_dir: str):

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    n_sigma = len(sigma_tab)
    n_alpha = len(alpha_tab)

    logger.info(f"Saving all figures in {output_dir}")

    for s in tqdm(range(n_sigma)):

        plt.figure()
        plt.scatter(alpha_tab, gen_grid[s, :])          
        plt.title(f'Generalization error for sigma = {sigma_tab[s]}')

        # Saving the figure
        fig_name = (f"sigma_{sigma_tab[s]}").replace(".","_")
        output_path = (output_dir / fig_name).with_suffix(".png")
        plt.savefig(str(output_path))
        plt.close()

    for a in tqdm(range(n_alpha)):

            plt.figure()
            plt.scatter(sigma_tab, gen_grid[:, a])
            plt.xscale("log")
            # plt.ylim(0., 100.)
            # plt.yscale("log")          
            plt.title(f'Generalization error for alpha = {alpha_tab[a]}')

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
        plt.title("Regression of alpha from the generalization bound")
        plt.savefig(str(alpha_reg_path))
        plt.close()



def analyze_one_seed(json_path: str):

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    num_exp = len(results.keys())
    logger.info(f"Found {num_exp} experiments")

    # Collect n_alpha and n_sigma
    n_sigma = 1 + max(results[k]["id_sigma"] for k in results.keys())
    n_alpha = 1 + max(results[k]["id_alpha"] for k in results.keys())

    # Collect sigma_tab, alpha_tab, and the generalization grids
    sigma_tab = np.zeros(n_sigma)
    sigma_factor_tab = np.zeros(n_sigma)
    alpha_tab = np.zeros(n_alpha)
    normalization_tab = np.zeros(n_alpha)
    acc_gen_grid = np.zeros((n_sigma, n_alpha))


    gen_tab = []
    bound_tab = []
    acc_bound_tab = []
    acc_tab = []
    sigma_values = []
    sigma_factor_values = []
    alpha_values = []
    

    # assert num_exp == n_sigma * n_alpha, (num_exp, n_sigma * n_alpha)

    for k in results.keys():

        # Estimate the actual value of the bound
        n = results[k]["n"]
        # normalization_factor = results[k]["normalization_factor"]
        alpha = results[k]["alpha"]
        n_params = results[k]["n_params"]
        sigma = results[k]["sigma"]  # true value, without normalization by the dim
        sigma_factor = results[k]["sigma"] * np.sqrt(n_params)
        gradient = results[k]["gradient_mean"]

        # TODO: this is ugly and suboptimal, find better
        sigma_tab[results[k]["id_sigma"]] = sigma
        sigma_factor_tab[results[k]["id_sigma"]] = sigma_factor
        alpha_tab[results[k]["id_alpha"]] = results[k]["alpha"]

        acc_gen_grid[
            results[k]["id_sigma"],
            results[k]["id_alpha"]
        ] = results[k]["acc_generalization"]

        # Collect generalization error ad sigma and alpha, for colored plots
        gen_tab.append(results[k]["loss_generalization"])
        acc_tab.append(results[k]["acc_generalization"])
        sigma_values.append(sigma)
        sigma_factor_values.append(sigma_factor)
        alpha_values.append(results[k]["alpha"])

        # in pytorch sgd, decay is the true decay, not post lr
        decay = results[k]["decay"]

        constant = asymptotic_constant(alpha, n_params)
        normalization_tab[results[k]["id_alpha"]] = constant

        bound_tab.append(np.sqrt(constant * gradient / (n * decay * np.power(sigma, alpha))))
        acc_bound_tab.append(np.sqrt(constant / (n * decay * np.power(sigma, alpha))))

    # Plot everything
    output_dir = json_path.parent.parent / (json_path.parent.stem + "_figures")
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving figures in {str(output_dir)}")

    plot_one_seed(acc_gen_grid, sigma_factor_tab, alpha_tab, str(output_dir))
    # plot_one_seed(acc_gen_grid / np.sqrt(normalization_tab[np.newaxis, :]), sigma_tab, alpha_tab, str(output_dir))
    plot_bound(gen_tab, bound_tab, output_dir, sigma_factor_values, alpha_values, log_scale=True)
    plot_bound(acc_tab, bound_tab, output_dir, sigma_factor_values, alpha_values, log_scale=True, stem="accuracy")
    

if __name__ == "__main__":
     fire.Fire(analyze_one_seed)










    

    