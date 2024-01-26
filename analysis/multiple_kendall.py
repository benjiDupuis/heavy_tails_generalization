import json
from pathlib import Path
from typing import List

import fire
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import kendalltau
from tqdm import tqdm

from analysis.kendall import granulated_kendalls_from_dict
from last_point.utils import matrix_robust_mean


def granulated_kendalls(json_path: str,
         generalization_key: str = "acc_generalization",
         complexity_keys: List[str] = ["alpha"],
         hyperparameters_keys: List[str] = ["n_params"]):
    """
    json_path should be all_results.json
    """

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    n_seed = len(list(results.keys()))
    logger.info(f"Found {n_seed} experiments")

    all_seeds_kendalls = {}
    final_results = {}

    for seed in tqdm(results.keys()):
        
        seed_results = results[seed]
        all_seeds_kendalls[seed] = granulated_kendalls_from_dict(seed_results,\
                                                            generalization_key, \
                                                            complexity_keys, \
                                                            hyperparameters_keys)

    final_results["granulated Kendalls"] = {}
    for comp in complexity_keys:
        final_results["granulated Kendalls"][comp] = {}
        # errors for each hyperparameter
        for hyp in hyperparameters_keys:
            final_results["granulated Kendalls"][comp][hyp] = {}
            coeff_list = []
            for seed in tqdm(all_seeds_kendalls.keys()):
                coeff_list.append(all_seeds_kendalls[seed]["granulated Kendalls"][comp][hyp])
            coeff_list = np.array(coeff_list)
            coeff_mean = coeff_list.mean()
            centered = coeff_list - coeff_mean
            coeff_dev = np.sqrt(np.power(centered, 2).sum() / (n_seed - 1))
            final_results["granulated Kendalls"][comp][hyp]["mean"] = coeff_mean
            final_results["granulated Kendalls"][comp][hyp]["dev"] = coeff_dev
        
        # Average coefficient
        final_results["granulated Kendalls"][comp]["average granulated Kendall coefficient"] = {}
        coeff_list = []
        for seed in tqdm(all_seeds_kendalls.keys()):
            coeff_list.append(all_seeds_kendalls[seed]["granulated Kendalls"][comp]["average granulated Kendall coefficient"])
        coeff_list = np.array(coeff_list)
        coeff_mean = coeff_list.mean()
        centered = coeff_list - coeff_mean
        coeff_dev = np.sqrt(np.power(centered, 2).sum() / (n_seed - 1))
        final_results["granulated Kendalls"][comp]["average granulated Kendall coefficient"]["mean"] = coeff_mean
        final_results["granulated Kendalls"][comp]["average granulated Kendall coefficient"]["dev"] = coeff_dev

        # Whole Kendall
        final_results["Kendall tau"] = {}
        final_results["Kendall tau"][comp] = {}
        coeff_list = []
        for seed in tqdm(all_seeds_kendalls.keys()):
            coeff_list.append(all_seeds_kendalls[seed]["Kendall tau"][comp])
        coeff_list = np.array(coeff_list)
        coeff_mean = coeff_list.mean()
        centered = coeff_list - coeff_mean
        coeff_dev = np.sqrt(np.power(centered, 2).sum() / (n_seed - 1))
        final_results["Kendall tau"][comp]["mean"] = coeff_mean
        final_results["Kendall tau"][comp]["dev"] = coeff_dev

    output_path = Path(json_path).parent / "multiple_kendalls.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results in {str(output_path)}")

    with open(str(output_path), "w") as output_file:
        json.dump(final_results, output_file, indent=2)

    logger.info(f"Results: \n {json.dumps(final_results, indent=2)}")

    return final_results
        

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
    # psi should be [alpha, seed]

    # psi_means = psi.mean(axis=1)
    # psi_deviations = np.sqrt(np.power((psi - psi_means[:, np.newaxis]), 2).sum(axis=1) / (n - 1))
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
    xlabel = r"Number of parameters $\mathbf{d}$" if key == "n_params" else r"$\mathbf{\sigma_1}$"
    plt.xlabel(xlabel, weight="bold")
    if key == "sigma":
        plt.xscale("log")
    plt.ylabel(r"Kendall $\mathbf{\tau}$ between accuracy gap and $\mathbf{\alpha}$", weight="bold")

    plt.plot(varying_tab, np.zeros(len(varying_tab)), "--", color="r")

    plt.grid()

    output_dir = json_path.parent / (json_path.parent.stem + "_figures")
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    fig_name = "alpha_correlation_with_errors"
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

    plt.legend()
    plt.savefig(str(output_path))
    plt.close()

    return varying_tab, psi_means, psi_deviations

    
def plot_alpha_kendall(json_path: str, key: str="n_params"):
    """
    dict is one seedor average_results
    """

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    varying_tab, kendall_tab = alpha_kendall(results, key=key)

    plt.figure()
    plt.scatter(varying_tab, kendall_tab)
    xlabel = "d" if key == "n_params" else "sigma"
    plt.xlabel(xlabel)
    if key == "sigma":
        plt.xscale("log")
    plt.ylabel(r"Kendall tau $\psi$")
    # plt.legend()
    # plt.title("Correlation gen/alpha with in function of sigma")

    output_dir = json_path.parent / (json_path.parent.stem + "_figures")
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving figures in {str(output_dir)}")

    fig_name = "alpha_correlation"
    output_path = (output_dir / fig_name).with_suffix(".png")

    plt.savefig(str(output_path))
    plt.close()


def plot_two_alpha_kendalls(json_1, json_2, key="sigma"):

    
    varying_tab_1, kendall_tab_1, dev_tab_1 = alpha_kendall_all_seeds(json_1, key=key)

    varying_tab_2, kendall_tab_2, dev_tab_2 = alpha_kendall_all_seeds(json_2, key=key)

    varying_tab = np.concatenate([varying_tab_1[1:], varying_tab_2[1:]])
    kendall_tab = np.concatenate([kendall_tab_1[1:], kendall_tab_2[1:]])
    dev_tab = np.concatenate([dev_tab_1[1:], dev_tab_2[1:]])

    plt.figure()
    plt.plot(varying_tab, kendall_tab, color = "g",label=r"$\psi$")
    plt.fill_between(varying_tab, \
                    kendall_tab - dev_tab,\
                    kendall_tab + dev_tab,
                    color = "g",
                    alpha = 0.25)
    xlabel = "d" if key == "n_params" else r"$\sigma$"
    plt.xlabel(xlabel)
    # plt.xscale("log")
    plt.ylabel("Kendall tau")
    plt.legend()

    output_dir = Path(".")

    fig_name = "alpha_correlation"
    output_path = (output_dir / fig_name).with_suffix(".png")

    plt.savefig(str(output_path))
    plt.close()




if __name__ == "__main__":
    # fire.Fire(plot_alpha_kendall)
    fire.Fire(alpha_kendall_all_seeds)
    # fire.Fire(plot_two_alpha_kendalls)




    







    






    







