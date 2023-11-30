import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm


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
        plt.title(f'Generalization for sigma = {sigma_tab[s]}')

        # Saving the figure
        fig_name = (f"sigma_{sigma_tab[s]}").replace(".","_")
        output_path = (output_dir / fig_name).with_suffix(".png")
        plt.savefig(str(output_path))
        plt.close()

    for a in tqdm(range(n_alpha)):

            plt.figure()
            plt.scatter(sigma_tab, gen_grid[:, a])
            plt.xscale("log")          
            plt.title(f'Generalization for alpha = {alpha_tab[a]}')

            # Saving the figure
            fig_name = (f"alpha_{alpha_tab[a]}").replace(".","_")
            output_path = (output_dir / fig_name).with_suffix(".png")
            plt.savefig(str(output_path))
            plt.close()


def analyze_one_seed(json_path: str):

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    num_exp = len(results.keys())
    logger.info(f"Found {num_exp} experiments")

    # Collect n_alpha and n_sigma
    n_sigma = max(results[k]["id_sigma"] for k in results.keys())
    n_alpha = max(results[k]["id_alpha"] for k in results.keys())

    # Collect sigma_tab, alpha_tab, and the generalization grids
    sigma_tab = np.zeros(n_sigma)
    alpha_tab = np.zeros(n_alpha)
    acc_gen_grid = np.zeros((n_sigma, n_alpha))

    assert num_exp == n_sigma * n_alpha, (num_exp, n_sigma * n_alpha)

    for k in results.keys():

        # TODO: this is ugly and suboptimal, find better
        sigma_tab[results[k]["id_sigma"]] = results[k]["sigma"]
        alpha_tab[results[k]["id_alpha"]] = results[k]["alpha"]

        acc_gen_grid[
            results[k]["id_sigma"],
            results[k]["id_alpha"]
        ] = results[k]["acc_generalization"]

    # Plot everything
    output_dir = json_path.parent
    plot_one_seed(acc_gen_grid, sigma_tab, alpha_tab, str(output_dir))
    

if __name__ == "__main__":
     fire.Fire(analyze_one_seed)










    

    