import json
from pathlib import Path
from typing import List

import fire
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import kendalltau
from tqdm import tqdm

from last_point.utils import matrix_robust_mean

font = {'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)


#########################
# This scripts are the ones we used to create the plots of the Kenall tau correlation coefficient
# of the accuracy error with respect to alpha
# The file they take as input, called results.json, collect the results of several experiments
#########################
      

def alpha_kendall(results: dict, key: str = "n_params", gen_key="acc_generalization"):
    """
    dict is one seed, e.g. average_results for instance
    """

    # We first collect, for each sigma, the alpha_tab and the gen_tab
    # and use them to compute the kendall coefficient

    if key == "sigma":
        varying = "sigma"
        varying_id = "id_sigma"
    elif key == "n_params":
        varying = "n_params"
        varying_id = "n_params"
    else:
        raise NotImplementedError()


    varying_tab = []
    kendall_tab = []
    fixed_results = {}

    for key in tqdm(results.keys()):

        if results[key]["alpha"] >= 1.6:

            if results[key][varying_id] not in fixed_results.keys():
                fixed_results[results[key][varying_id]] = {
                    "alpha": [],
                    "gen": []
                }

            fixed_results[results[key][varying_id]]["alpha"].append(
                results[key]["alpha"]
            )
            fixed_results[results[key][varying_id]]["gen"].append(
                results[key][gen_key]
            )
            fixed_results[results[key][varying_id]][varying] = results[key][varying]

    for v in tqdm(fixed_results.keys()):

        varying_tab.append(fixed_results[v][varying])
        kendall_tab.append(kendalltau(
            fixed_results[v]["alpha"],
            fixed_results[v]["gen"]
        ).correlation)


    return np.array(varying_tab), np.array(kendall_tab)


def alpha_kendall_all_seeds(json_path: str, key: str = "n_params", av_path: str = None):
    """
    json_path should be all_results.json
    """

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    kendalls = []

    for seed in tqdm(results.keys()):
        varying_tab, kendall_tab = alpha_kendall(results[seed], key=key)
        kendalls.append(kendall_tab[:, np.newaxis])
    
    psi = np.concatenate(kendalls, axis=1)
    assert psi.shape[0] == len(varying_tab)
    assert psi.shape[1] == len(results.keys())

    n = len(results.keys())
    psi_means, psi_deviations = matrix_robust_mean(psi, quantile=0.)

    # we order varying_tab
    indices = np.argsort(varying_tab)
    varying_tab = varying_tab[indices]
    psi_means = psi_means[indices]
    psi_deviations = psi_deviations[indices]
    
    # Plots
    plt.figure()
    plt.fill_between(varying_tab, \
                    psi_means - psi_deviations,\
                    psi_means + psi_deviations,
                    color = "g",
                    alpha = 0.25)
    plt.plot(varying_tab, psi_means, color = "g",label=r"$\mathbf{\tau}$")
    xlabel = r"Number of parameters $\mathbf{d}$" if key == "n_params" else r"$\mathbf{\sigma_1}$"
    plt.xlabel(xlabel, weight="bold", fontsize=15)
    if key == "sigma":
        plt.xscale("log")
    plt.ylabel(r"Kendall $\mathbf{\tau}$ accuracy gap vs. $\mathbf{\alpha}$", weight="bold", fontsize=15)

    plt.plot(varying_tab, np.zeros(len(varying_tab)), "--", color="r")

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    fig_name = "alpha_correlation_with_errors_no_mean"
    output_path = (output_dir / fig_name).with_suffix(".png")

    logger.info(f"Saving figures in {str(output_path)}")

    if av_path is not None:
        
        av_path = Path(av_path)
        assert av_path.exists(), str(av_path)

        with open(str(av_path), "r") as json_file:
            results = json.load(json_file)

        varying_tab, kendall_tab = alpha_kendall(results,\
                                                 key=key)

        # we order varying_tab
        indices = np.argsort(varying_tab)
        varying_tab = varying_tab[indices]
        kendall_tab = kendall_tab[indices]

        plt.plot(varying_tab, kendall_tab, "--", color="k",\
                    label=r"$\mathbf{\tau}$ of the mean generalization error wrt $\alpha$")

    plt.ticklabel_format(axis='x')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(str(output_path), bbox_inches="tight")

    output_path = (output_dir / fig_name).with_suffix(".pdf")
    logger.info(f"Saving figures in {str(output_path)}")
    plt.savefig(str(output_path), bbox_inches="tight")

    plt.close()

    return varying_tab, psi_means, psi_deviations

    
def plot_alpha_kendall(json_path: str, key: str="n_params"):
    """
    dict is one seed or average_results
    """

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    varying_tab, kendall_tab = alpha_kendall(results, key=key)

    indices = np.argsort(varying_tab)
    varying_tab = varying_tab[indices]
    kendall_tab = kendall_tab[indices]

    plt.figure()

    plt.plot(varying_tab, kendall_tab, "--x", color="k",\
                    label=r"$\mathbf{\tau}$ of the mean generalization error wrt $\alpha$")

    xlabel = "d" if key == "n_params" else "sigma"
    plt.xlabel(xlabel)
    if key == "sigma":
        plt.xscale("log")
    plt.ylabel(r"Kendall tau $\psi$")
    
    output_dir = json_path.parent / (json_path.parent.stem + "_figures")
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving figures in {str(output_dir)}")

    fig_name = "alpha_correlation"
    output_path = (output_dir / fig_name).with_suffix(".png")

    plt.plot(varying_tab, np.zeros(len(varying_tab)), "--", color="r")

    plt.grid()
    plt.legend()

    plt.savefig(str(output_path))
    plt.close()




if __name__ == "__main__":
    fire.Fire(alpha_kendall_all_seeds)




    







    






    







