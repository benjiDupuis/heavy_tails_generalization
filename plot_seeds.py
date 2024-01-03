import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from loguru import logger
from tqdm import tqdm

from last_point.utils import linear_regression, all_linear_regression
from last_point.simulation import asymptotic_constant
from last_point.utils import poly_alpha

def all_alpha_regression(json_path: str):
    """
    json_path shoul be an all_results file
    """
    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    num_exp = len(results.keys())
    logger.info(f"Found {num_exp} experiments")

    n_seed = len(list(results.keys()))
    seed_list = list(results.keys())
    logger.info(f"Found {n_seed} seeds")

    key_example = list(results.keys())[0]
    n_alpha = 1 + max(results[key_example][k]["id_alpha"] for k in results[key_example].keys())

    n_params = results[key_example]["0"]["n_params"]

    values = {a:{"sigma": [], "gen": []} for a in range(n_alpha)}

    for seed_id in tqdm(range(n_seed)):
        
        seed = seed_list[seed_id]
        seed_results = results[seed]

        for k in seed_results.keys():

            alpha_id = seed_results[k]["id_alpha"]
            values[alpha_id]["alpha"] = seed_results[k]["alpha"]
            values[alpha_id]["sigma"].append(seed_results[k]["sigma"])
            values[alpha_id]["gen"].append(seed_results[k]["acc_generalization"])

    alpha_tab = np.zeros(n_alpha)
    alpha_est_tab = np.zeros(n_alpha)
    for a in values.keys():

        sigma_tab = np.array(values[a]["sigma"])
        gen_tab = np.array(values[a]["gen"])
        alpha_tab[a] = values[a]["alpha"]

        indices = (gen_tab > 0.) * (sigma_tab > 2./np.sqrt(n_params))
        alpha_est_tab[a] = 2. * linear_regression(np.log(1./sigma_tab[indices]),
                                            np.log(gen_tab[indices]))

    if all(alpha_est_tab[k] is not None for k in range(len(alpha_tab))):
        alpha_reg_path = (output_dir / "alpha_regression_all_data").with_suffix(".png")
        logger.info(f"Saving regression figure in {str(alpha_reg_path)}")

        plt.figure()
        ref_alpha = np.linspace(np.min(alpha_tab), np.max(alpha_tab), 100)
        plt.plot(ref_alpha, ref_alpha, color = "r")
        plt.scatter(alpha_tab, alpha_est_tab)
        plt.title("Regression of alpha from the generalization bound")
        plt.savefig(str(alpha_reg_path))
        plt.close()

    
def regressions_several_seeds(json_path: str):
    """
    json_path shoul be an all_results file
    """
    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    num_exp = len(results.keys())
    logger.info(f"Found {num_exp} experiments")

    n_seed = len(list(results.keys()))
    seed_list = list(results.keys())
    logger.info(f"Found {n_seed} seeds")

    key_example = list(results.keys())[0]
    n_alpha = 1 + max(results[key_example][k]["id_alpha"] for k in results[key_example].keys())

    n_params = results[key_example]["0"]["n_params"]

    alpha_regressions = np.zeros((n_alpha, n_seed))

    for seed_id in tqdm(range(n_seed)):
        
        seed = seed_list[seed_id]
        seed_results = results[seed]

        n_sigma = 1 + max(seed_results[k]["id_sigma"] for k in seed_results.keys())
        assert n_alpha == n_sigma, (n_alpha, n_sigma)

        acc_gen_grid = np.zeros((n_sigma, n_alpha))
        gradient_grid = np.zeros((n_sigma, n_alpha))
        sigma_tab = np.zeros(n_sigma)
        alpha_tab = np.zeros(n_alpha)

        for k in seed_results.keys():

            sigma_tab[seed_results[k]["id_sigma"]] = seed_results[k]["sigma"]
            alpha_tab[seed_results[k]["id_alpha"]] = seed_results[k]["alpha"]

            acc_gen_grid[
                    seed_results[k]["id_sigma"],
                    seed_results[k]["id_alpha"]
                ] = seed_results[k]["acc_generalization"]
            
            gradient_grid[
                    seed_results[k]["id_sigma"],
                    seed_results[k]["id_alpha"]
                ] = seed_results[k]["gradient_mean"]

        alpha_reg, _ = all_linear_regression(
                                            acc_gen_grid,
                                            sigma_tab,
                                            alpha_tab,
                                            sigma_low = 1./np.sqrt(n_params)
                                            )

        alpha_regressions[:, seed_id] = alpha_reg

    alpha_means = alpha_regressions.mean(axis=1)
    centered = alpha_regressions - alpha_means[:, np.newaxis]
    alpha_deviations = 0.5 * np.sqrt(np.power(centered, 2).sum(axis=1) / (n_alpha - 1))

    if all(alpha_means[k] is not None for k in range(len(alpha_tab))):
        alpha_reg_path = (output_dir / "alpha_regressions").with_suffix(".png")
        logger.info(f"Saving regression figure in {str(alpha_reg_path)}")

        plt.figure()
        plt.plot(np.linspace(1.7,2, 100), np.linspace(1.7,2,100), color = "r")
        plt.errorbar(alpha_tab, alpha_means, yerr=alpha_deviations, fmt="x")
        plt.title("Regression of alpha from the generalization bound")
        plt.savefig(str(alpha_reg_path))
        plt.close()


if __name__ =="__main__":
    fire.Fire(all_alpha_regression)






