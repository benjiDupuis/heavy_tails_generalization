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


def main(json_path: str,
         generalization_key: str = "acc_generalization",
         complexity_keys: List[str] = ["estimated_bound"],
         hyperparameters_keys: List[str] = ["sigma", "alpha"]):
    """
    json_path should be all results
    """

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    n_seed = len(list(results.keys()))
    logger.info(f"Found {n_seed} experiments")

    all_seeds_kendalls = {}
    final_results = {}

    for seed_id in tqdm(range(n_seed)):
        
        seed = seed_list[seed_id]
        seed_results = results[seed]

        all_seeds_kendalls[seed] = granulated_kendalls_from_dict(seed_results)

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
            coeff_dev = no.sqrt(np.power(centered, 2).sum() / (n_seed - 1))
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
        coeff_dev = no.sqrt(np.power(centered, 2).sum() / (n_seed - 1))
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
        coeff_dev = no.sqrt(np.power(centered, 2).sum() / (n_seed - 1))
        final_results["Kendall tau"][comp]["mean"] = coeff_mean
        final_results["Kendall tau"][comp]["dev"] = coeff_dev

    output_path = Path(json_path).parent / "multiple_kendalls.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results in {str(output_path)}")

    with open(str(output_path), "w") as output_file:
        json.dump(final_results, output_file, indent=2)

    logger.info(f"Results: \n {json.dumps(final_results, indent=2)}")

    return final_results

        


def alpha_kendall(results: dict):
    """
    dict is one seed, e.g. average_results for instance
    """

    # We first collect, for each sigma, the alpha_tab and the gen_tab
    # and use them to compute the kendall coefficient
    n_sigma = 1 + max(results[k]["id_sigma"] for k in results.keys())

    sigma_tab = []
    kendall_tab = []

    fixed_sigma_results = {k:{} for k in range(n_sigma)}
    for k in range(n_sigma):
        fixed_sigma_results[k]["alpha"] = []
        fixed_sigma_results[k]["gen"] = []

    for key in tqdm(results.keys()):

        fixed_sigma_results[results[key]["id_sigma"]]["alpha"].append(
            results[key]["alpha"]
        )
        fixed_sigma_results[results[key]["id_sigma"]]["gen"].append(
            results[key]["acc_gen_normalized"]
        )
        fixed_sigma_results[results[key]["id_sigma"]]["sigma"] = results[key]["sigma"]

    for sigma_id in tqdm(fixed_sigma_results.keys()):

        sigma_tab.append(fixed_sigma_results[sigma_id]["sigma"])
        kendall_tab.append(kendalltau(
            fixed_sigma_results[sigma_id]["alpha"],
            fixed_sigma_results[sigma_id]["gen"]
        ).correlation)

    return sigma_tab, kendall_tab

def plot_alpha_kendall(json_path: str):
    """
    dict is one seed, e.g. average_results for instance
    """

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    sigma_tab, kendall_tab = alpha_kendall(results)

    plt.figure()
    plt.scatter(sigma_tab, kendall_tab)
    plt.xlabel("sigma")
    plt.xscale("log")
    plt.ylabel("Kendall tau")
    plt.title("Correlation gen/alpha with in function of sigma")

    output_dir = json_path.parent.parent / (json_path.parent.stem + "_figures")
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving figures in {str(output_dir)}")

    fig_name = "alpha_correlation"
    output_path = (output_dir / fig_name).with_suffix(".png")

    plt.savefig(str(output_path))
    plt.close()


if __name__ == "__main__":
    fire.Fire(plot_alpha_kendall)




    







    






    







