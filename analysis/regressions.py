import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm

from last_point.utils import linear_regression

   
def regressions_several_seeds_dim(json_path: str):
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

    n_seed = len(list(results.keys()))
    seed_list = list(results.keys())
    logger.info(f"Found {n_seed} seeds")

    key_example = list(results.keys())[0]
    n_alpha = 1 + max(results[key_example][k]["id_alpha"] for k in results[key_example].keys())

    alpha_regressions = np.zeros((n_alpha, n_seed))
    alpha_regressions_corrected = np.zeros((n_alpha, n_seed))
    alpha_tab = np.zeros(n_alpha)

    for seed_id in tqdm(range(n_seed)):
        
        seed = seed_list[seed_id]
        seed_results = results[seed]

        n_width = 1 + max(seed_results[k]["id_sigma"] for k in seed_results.keys())

        values = {}

        for k in seed_results.keys():

            if seed_results[k]["id_alpha"] not in values.keys():
                values[seed_results[k]["id_alpha"]] = {"n_params": [], "gen": [], "gen_corrected": []}

            alpha_id = seed_results[k]["id_alpha"]
            values[alpha_id]["alpha"] = seed_results[k]["alpha"]
            values[alpha_id]["n_params"].append(seed_results[k]["n_params"])
            values[alpha_id]["gen"].append(seed_results[k]["acc_generalization"])


        for a in values.keys():

            params_tab = np.array(values[a]["n_params"])
            gen_tab = np.array(values[a]["gen"])
            gen_corrected_tab = np.array(values[a]["gen_corrected"])
            alpha_tab[a] = values[a]["alpha"]

            # We remove potential negative values of the generalization error, as we take the log in our regression
            indices = (gen_tab > 0.)

            if len(np.where(indices==1)[0]) > 1:
                reg = linear_regression(np.log(params_tab[indices]),
                                                    np.log(gen_tab[indices]))
                # reg = regression_selection(params_tab[indices], gen_tab[indices])
                # logger.info(reg)
                alpha_regressions[a, seed_id] = 2. - 4. * reg      
            else:
                logger.debug(f"No indices for alpha={alpha_tab[a]}")
                alpha_regressions[a, seed_id] = None

    alpha_means = alpha_regressions.mean(axis=1)
    centered = alpha_regressions - alpha_means[:, np.newaxis]
    alpha_deviations = np.sqrt(np.power(centered, 2).sum(axis=1) / (n_seed - 1))

    if all(alpha_means[k] is not None for k in range(len(alpha_tab))):
        alpha_reg_path = (output_dir / "alpha_regressions_from_n_params").with_suffix(".png")
        logger.info(f"Saving regression figure in {str(alpha_reg_path)}")

        plt.figure()
        alphas_gt = np.linspace(np.min(alpha_tab), np.max(alpha_tab))
        plt.fill_between(alpha_tab, \
                            alpha_means - alpha_deviations,\
                             alpha_means + alpha_deviations,
                             color = "b",
                             alpha = 0.25)
        plt.plot(alphas_gt, alphas_gt, color = "r", label=r"Ground truth $\alpha$")
        plt.plot(alpha_tab, alpha_means, color = "b",label=r"Estimated $\hat{\alpha}$")
        plt.ylabel(r"Estimated tail index $\mathbf{\hat{\alpha}}$", weight="bold", fontsize=15)
        plt.xlabel(r"Tail index $\mathbf{\alpha}$", weight="bold", fontsize=15)
        plt.grid()
        plt.legend()

        plt.savefig(str(alpha_reg_path))
        plt.close()



def plot_generalization_against_d(json_path: str):

    """
    json_path shoul be an average_results file
    """
    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    output_dir = json_path.parent / "figures"
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    n_exp = len(results.keys())
    logger.info(f"Found {n_exp} experiments")

    alpha_values = {}

    for key in results.keys():

        a = results[key]["id_alpha"]
        alpha = results[key]["alpha"]

        if alpha >= 1.7:

            if a not in alpha_values.keys():
                alpha_values[a] = {
                    "alpha": alpha,
                    "gen": [],
                    "d": []
                }

            alpha_values[a]["alpha"] = alpha
            alpha_values[a]["gen"].append(results[key]["acc_generalization"])
            alpha_values[a]["d"].append(results[key]["n_params"])

    plt.figure()

    for alpha_id in alpha_values.keys():

        indices = np.argsort(alpha_values[alpha_id]["d"])
        alpha = alpha_values[alpha_id]["alpha"]
        plt.loglog(np.array(alpha_values[alpha_id]["d"])[indices],
                    np.array(alpha_values[alpha_id]["gen"])[indices],
                    "--x",
                    label=f"alpha: {round(alpha, 2)}")

    plt.grid(visible=True, which="minor")
    plt.legend()

    loglog_path = (output_dir / "log_gen_vs_log_d").with_suffix(".png")
    logger.info(f"Saving regression figure in {str(loglog_path)}")

    plt.savefig(str(loglog_path))


if __name__ =="__main__":
    fire.Fire(regressions_several_seeds_dim)
    # fire.Fire(plot_generalization_against_d)






