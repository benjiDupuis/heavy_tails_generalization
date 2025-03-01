import json
from pathlib import Path
from typing import List

import fire
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import pearsonr
from tqdm import tqdm

from last_point.utils import matrix_robust_mean


#########################
# This scripts are the ones we used to create the plots of the Kenall tau correlation coefficient
# of the accuracy error with respect to alpha
# The file they take as input, called results.json, collect the results of several experiments
#########################
      

def alpha_pearson(results: dict, key: str = "n_params"):
    """
    dict is one seed, e.g. average_results for instance
    """

    # We first collect, for each sigma, the alpha_tab and the gen_tab
    # and use them to compute the pearson coefficient

    if key == "sigma":
        varying = "sigma"
        varying_id = "id_sigma"
    elif key == "n_params":
        varying = "n_params"
        varying_id = "n_params"
    else:
        raise NotImplementedError()


    varying_tab = []
    pearson_tab = []
    fixed_results = {}

    for key in tqdm(results.keys()):

        if results[key][varying_id] not in fixed_results.keys():
            fixed_results[results[key][varying_id]] = {
                "alpha": [],
                "gen": []
            }

        fixed_results[results[key][varying_id]]["alpha"].append(
            results[key]["alpha"]
        )
        fixed_results[results[key][varying_id]]["gen"].append(
            results[key]["acc_generalization"]
        )
        fixed_results[results[key][varying_id]][varying] = results[key][varying]

    for v in tqdm(fixed_results.keys()):

        varying_tab.append(fixed_results[v][varying])
        pearson_tab.append(pearsonr(
            fixed_results[v]["alpha"],
            fixed_results[v]["gen"]
        ).correlation)


    return np.array(varying_tab), np.array(pearson_tab)


def alpha_pearson_all_seeds(json_path: str, key: str = "n_params", av_path: str = None):
    """
    json_path should be all_results.json
    """

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    pearsons = []

    for seed in tqdm(results.keys()):
        varying_tab, pearson_tab = alpha_pearson(results[seed], key=key)
        pearsons.append(pearson_tab[:, np.newaxis])
    
    psi = np.concatenate(pearsons, axis=1)
    assert psi.shape[0] == len(varying_tab)
    assert psi.shape[1] == len(results.keys())

    n = len(results.keys())
    psi_means, psi_deviations = matrix_robust_mean(psi)

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
    xlabel = "d" if key == "n_params" else r"$\sigma$"
    plt.xlabel(xlabel, weight="bold")
    if key == "sigma":
        plt.xscale("log")
    plt.ylabel(r"pearson $\mathbf{\tau}$", weight="bold")

    plt.plot(varying_tab, np.zeros(len(varying_tab)), "--", color="r")

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    fig_name = "alpha_correlation_with_errors_pearson"
    output_path = (output_dir / fig_name).with_suffix(".png")

    logger.info(f"Saving figures in {str(output_path)}")

    if av_path is not None:
        
        av_path = Path(av_path)
        assert av_path.exists(), str(av_path)

        with open(str(av_path), "r") as json_file:
            results = json.load(json_file)

        varying_tab, pearson_tab = alpha_pearson(results, key=key)

        # we order varying_tab
        indices = np.argsort(varying_tab)
        varying_tab = varying_tab[indices]
        pearson_tab = pearson_tab[indices]

        plt.plot(varying_tab, pearson_tab, "--", color="k",\
                    label=r"$\mathbf{\tau}$ of the mean generalization error wrt $\alpha$")

    plt.legend()
    plt.grid()
    plt.savefig(str(output_path))
    plt.close()

    return varying_tab, psi_means, psi_deviations

    
def plot_alpha_pearson(json_path: str, key: str="n_params"):
    """
    dict is one seedor average_results
    """

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    varying_tab, pearson_tab = alpha_pearson(results, key=key)

    plt.figure()
    plt.scatter(varying_tab, pearson_tab)
    xlabel = "d" if key == "n_params" else "sigma"
    plt.xlabel(xlabel)
    if key == "sigma":
        plt.xscale("log")
    plt.ylabel(r"pearson tau $\psi$")
    
    output_dir = json_path.parent / (json_path.parent.stem + "_figures")
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving figures in {str(output_dir)}")

    fig_name = "alpha_correlation"
    output_path = (output_dir / fig_name).with_suffix(".png")

    plt.savefig(str(output_path))
    plt.close()




if __name__ == "__main__":
    fire.Fire(alpha_pearson_all_seeds)
    # fire.Fire(plot_alpha_pearson)





    







    






    







