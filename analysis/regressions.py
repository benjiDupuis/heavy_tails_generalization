import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from loguru import logger
from tqdm import tqdm

from last_point.utils import linear_regression, all_linear_regression, regression_selection
from last_point.simulation import asymptotic_constant
from last_point.utils import poly_alpha, matrix_robust_mean

from matplotlib import rc, rcParams
rcParams['font.weight'] = 'bold'


def dimension_regressions(json_path: str):
    """
    This scipts takes an all_results.json file and returns the regression
    of alpha based on n_params
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

    key_example = list(results.keys())[0]
    n_alpha = 1 + max(results[key_example][k]["id_alpha"] for k in results[key_example].keys())
    n_seed = len(list(results.keys()))

    values = {a:{"n_params": [], "gen": []} for a in range(n_alpha)}

    for seed in tqdm(results.keys()):
        
        seed_results = results[seed]

        for k in seed_results.keys():

            alpha_id = seed_results[k]["id_alpha"]
            values[alpha_id]["alpha"] = seed_results[k]["alpha"]
            values[alpha_id]["n_params"].append(seed_results[k]["n_params"])
            # Be aware that we use the normalized accuracy error
            values[alpha_id]["gen"].append(seed_results[k]["acc_generalization"])

    alpha_tab = np.zeros(n_alpha)
    alpha_est_tab = np.zeros(n_alpha)
    for a in values.keys():

        params_tab = np.array(values[a]["n_params"])
        gen_tab = np.array(values[a]["gen"])
        alpha_tab[a] = values[a]["alpha"]

        indices = (gen_tab > 0.) * (params_tab > np.quantile(params_tab, 0.1))

        if len(indices) > 1:
            reg = linear_regression(np.log(params_tab[indices]),
                                                np.log(gen_tab[indices]))
            alpha_est_tab[a] = 2. - 4. * reg      
        else:
            logger.debug(f"No indices for alpha={alpha_tab[a]}")
            alpha_est_tab[a] = None

    if all(alpha_est_tab[k] is not None for k in range(len(alpha_tab))):
        alpha_reg_path = (output_dir / "alpha_regression_from_n_params").with_suffix(".png")
        logger.info(f"Saving regression figure in {str(alpha_reg_path)}")

        plt.figure()
        ref_alpha = np.linspace(np.min(alpha_tab), np.max(alpha_tab), 100)
        plt.plot(ref_alpha, ref_alpha, color = "r")
        plt.scatter(alpha_tab, alpha_est_tab)
        plt.title("Regression of alpha from the generalization bound")
        plt.savefig(str(alpha_reg_path))
        plt.close()




def sigma_regression(json_path: str):
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

        indices = (gen_tab > 0.) * (sigma_tab > 1./np.sqrt(n_params))
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

    
def regressions_several_seeds_sigma(json_path: str):
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
    alpha_tab = np.zeros(n_alpha)


    for seed_id in tqdm(range(n_seed)):
        
        seed = seed_list[seed_id]
        seed_results = results[seed]

        values = {a:{"sigma": [], "gen": []} for a in range(n_alpha)}

        for k in seed_results.keys():

            alpha_id = seed_results[k]["id_alpha"]
            values[alpha_id]["alpha"] = seed_results[k]["alpha"]
            values[alpha_id]["sigma"].append(seed_results[k]["sigma"])
            # Be aware that we use the normalized accuracy error
            values[alpha_id]["gen"].append(seed_results[k]["acc_generalization"])
            # values[alpha_id]["gen"].append(seed_results[k]["acc_generalization"] / np.sqrt(seed_results[k]["gradient_mean"]))

        for a in values.keys():

            sigma_tab = np.array(values[a]["sigma"])
            gen_tab = np.array(values[a]["gen"])
            alpha_tab[a] = values[a]["alpha"]

            # indices = (gen_tab > 0.) * (sigma_tab > np.quantile(sigma_tab, 0.5))
            indices = (gen_tab > 0.)

            if len(np.where(indices == 1)[0]) > 1:
                reg = linear_regression(np.log(1./sigma_tab[indices]),
                                                    np.log(gen_tab[indices]))
                alpha_regressions[a, seed_id] = 2. * reg      
            else:
                logger.debug(f"No indices for alpha={alpha_tab[a]}")
                alpha_regressions[a, seed_id] = None

    alpha_means = alpha_regressions.mean(axis=1)
    # centered = alpha_regressions - alpha_means[:, np.newaxis]
    # alpha_deviations = np.sqrt(np.power(centered, 2).sum(axis=1) / (n_alpha - 1))
    alpha_means, alpha_deviations = matrix_robust_mean(alpha_regressions)

    # HACK
    ids = np.argwhere(~np.isnan(alpha_means))[:, 0]

    alpha_means = alpha_means[ids]
    alpha_deviations = alpha_deviations[ids]
    alpha_tab = alpha_tab[ids]

    alpha_reg_path = (output_dir / "alpha_regressions_from_sigma").with_suffix(".png")
    logger.info(f"Saving regression figure in {str(alpha_reg_path)}")

    plt.figure()
    alphas_gt = np.linspace(np.min(alpha_tab), np.max(alpha_tab))
    plt.fill_between(alpha_tab, \
                        alpha_means - alpha_deviations,\
                            alpha_means + alpha_deviations,
                            color = "b",
                            alpha = 0.25)
    plt.plot(alphas_gt, alphas_gt, color = "r", label=r"Ground truth $\alpha$")
    # plt.errorbar(alpha_tab, alpha_means, yerr=alpha_deviations, fmt="x")
    plt.plot(alpha_tab, alpha_means, color = "b",label=r"Estimated $\hat{\alpha}$")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\hat{\alpha}$")
    plt.legend()
    # plt.plot(alphas_gt, alphas_gt, color = "r")
    # plt.errorbar(alpha_tab, alpha_means, yerr=alpha_deviations, fmt="x")
    # plt.title("Regression of alpha from the generalization bound")
    plt.savefig(str(alpha_reg_path))
    plt.close()


   
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

        # values = {a:{"n_params": [], "gen": []} for a in range(n_alpha)}
        values = {}

        for k in seed_results.keys():

            if seed_results[k]["id_alpha"] not in values.keys():
                values[seed_results[k]["id_alpha"]] = {"n_params": [], "gen": [], "gen_corrected": []}

            alpha_id = seed_results[k]["id_alpha"]
            values[alpha_id]["alpha"] = seed_results[k]["alpha"]
            values[alpha_id]["n_params"].append(seed_results[k]["n_params"])
            # Be aware that we use the normalized accuracy error
            values[alpha_id]["gen"].append(seed_results[k]["acc_generalization"])
            values[alpha_id]["gen_corrected"].append(seed_results[k]["acc_generalization"] / np.sqrt(seed_results[k]["gradient_mean"]))


        for a in values.keys():

            params_tab = np.array(values[a]["n_params"])
            gen_tab = np.array(values[a]["gen"])
            gen_corrected_tab = np.array(values[a]["gen_corrected"])
            alpha_tab[a] = values[a]["alpha"]

            indices = (gen_tab > 0.) 
            # indices = (gen_tab > np.quantile(gen_tab, 0.1)) * (gen_tab < np.quantile(gen_tab, 0.9))

            if len(np.where(indices==1)[0]) > 1:
                reg = linear_regression(np.log(params_tab[indices]),
                                                    np.log(gen_tab[indices]))
                # reg = regression_selection(params_tab[indices], gen_tab[indices])
                # logger.info(reg)
                alpha_regressions[a, seed_id] = 2. - 4. * reg      
            else:
                logger.debug(f"No indices for alpha={alpha_tab[a]}")
                alpha_regressions[a, seed_id] = None

            if len(np.where(indices==1)[0]) > 1:
                # reg = regression_selection(params_tab[indices], gen_corrected_tab[indices])

                reg = linear_regression(np.log(params_tab[indices]),
                                                    np.log(gen_corrected_tab[indices]))
                alpha_regressions_corrected[a, seed_id] = 2. - 4. * reg      
            else:
                logger.debug(f"No indices for alpha={alpha_tab[a]}")
                alpha_regressions[a, seed_id] = None

    # alpha_means = alpha_regressions.mean(axis=1)
    # centered = alpha_regressions - alpha_means[:, np.newaxis]
    # alpha_deviations = np.sqrt(np.power(centered, 2).sum(axis=1) / (n_seed - 1))

    # alpha_means_corrected = alpha_regressions_corrected.mean(axis=1)
    # centered = alpha_regressions_corrected - alpha_means_corrected[:, np.newaxis]
    # alpha_deviations_corrected = np.sqrt(np.power(centered, 2).sum(axis=1) / (n_seed - 1))

    alpha_means, alpha_deviations = matrix_robust_mean(alpha_regressions)
    alpha_means_corrected, alpha_deviations_corrected = matrix_robust_mean(alpha_regressions_corrected)

    if all(alpha_means[k] is not None for k in range(len(alpha_tab))):
        alpha_reg_path = (output_dir / "alpha_regressions_from_n_params").with_suffix(".png")
        logger.info(f"Saving regression figure in {str(alpha_reg_path)}")

        plt.figure(figsize=(9,5))
        # plt.ylim(0., min(4., np.max(alpha_means_corrected + alpha_deviations_corrected)))
        alphas_gt = np.linspace(np.min(alpha_tab), np.max(alpha_tab))
        plt.fill_between(alpha_tab, \
                            alpha_means - alpha_deviations,\
                             alpha_means + alpha_deviations,
                             color = "b",
                             alpha = 0.25)
        # plt.plot(alphas_gt, alphas_gt, color = "r", label=r"Ground truth $\alpha$")
        # plt.errorbar(alpha_tab, alpha_means, yerr=alpha_deviations, fmt="x")
        plt.plot(alpha_tab, alpha_means, color = "b",label=r"Estimated $\hat{\alpha}$")
        plt.ylabel(r"Estimated tail index $\mathbf{\hat{\alpha}}$", weight="bold", fontsize=15)
        plt.xlabel(r"Tail index $\mathbf{\alpha}$", weight="bold", fontsize=15)
        plt.grid()
        plt.legend()

        # Uncomment to plot the gradient corrected version
        # plt.fill_between(alpha_tab, \
        #                     alpha_means_corrected - alpha_deviations_corrected,\
        #                      alpha_means_corrected + alpha_deviations_corrected,
        #                      color = "gray",
        #                      alpha = 0.25)
        # plt.plot(alpha_tab, alpha_means_corrected, color = "k",label=r"Estimated $\hat{\alpha}$ with gradient correction")
        plt.legend()

        plt.savefig(str(alpha_reg_path))
        plt.close()


def main(json_path: str, mode: str = "regressions_several_seeds_dim"):

    if mode == "dimension_regressions":
        dimension_regressions(json_path)
    elif mode == "sigma_regression":
        sigma_regression(json_path)
    elif mode == "regressions_several_seeds_sigma":
        regressions_several_seeds_sigma(json_path)
    elif mode == "regressions_several_seeds_dim":
        regressions_several_seeds_dim(json_path)
    else:
        raise NotImplementedError()



if __name__ =="__main__":
    fire.Fire(main)






