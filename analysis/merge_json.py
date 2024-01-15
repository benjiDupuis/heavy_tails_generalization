import json
from pathlib import Path

import fire
import numpy as np
from loguru import logger
from tqdm import tqdm

from last_point.utils import robust_mean


def average_results(all_results: dict) -> dict:
    """
    all_results contains the results of several dicts
    results are collected based on id_sigma and id_alpha
    """
    
    # we first construct a dict of lists
    dict_of_lists = {}

    for key_seed in tqdm(all_results.keys()):
        for key_exp in tqdm(all_results[key_seed].keys()):

            # we construct the new key
            key_id = "_".join([
                str(all_results[key_seed][key_exp]["id_sigma"]),
                str(all_results[key_seed][key_exp]["id_alpha"]),
                str(all_results[key_seed][key_exp]["width"])
            ])

            # creating the dict and adding the elements
            if key_id not in dict_of_lists.keys():
                dict_of_lists[key_id] = {}

            # All the following should not change with the seed
            dict_of_lists[key_id]["alpha"] = all_results[key_seed][key_exp]["alpha"]
            dict_of_lists[key_id]["sigma"] = all_results[key_seed][key_exp]["sigma"]
            dict_of_lists[key_id]["horizon"] = all_results[key_seed][key_exp]["horizon"]
            dict_of_lists[key_id]["eta"] = all_results[key_seed][key_exp]["eta"]
            dict_of_lists[key_id]["id_sigma"] = all_results[key_seed][key_exp]["id_sigma"]
            dict_of_lists[key_id]["id_alpha"] = all_results[key_seed][key_exp]["id_alpha"]
            dict_of_lists[key_id]["eta"] = all_results[key_seed][key_exp]["eta"]
            dict_of_lists[key_id]["decay"] = all_results[key_seed][key_exp]["decay"]
            dict_of_lists[key_id]["K_constant"] = all_results[key_seed][key_exp]["K_constant"]
            dict_of_lists[key_id]["n_params"] = all_results[key_seed][key_exp]["n_params"]
            dict_of_lists[key_id]["n"] = all_results[key_seed][key_exp]["n"]
            dict_of_lists[key_id]["n_val"] = all_results[key_seed][key_exp]["n_val"]
            dict_of_lists[key_id]["n_ergodic"] = all_results[key_seed][key_exp]["n_ergodic"]
            dict_of_lists[key_id]["acc_gen_normalized"] = all_results[key_seed][key_exp]["acc_gen_normalized"]

            # HACK improve it
            if "acc_generalization" not in dict_of_lists[key_id].keys():
                dict_of_lists[key_id]["acc_generalization"] = []
            dict_of_lists[key_id]["acc_generalization"].append(all_results[key_seed][key_exp]["acc_generalization"])

            # if "acc_gen_normalized" not in dict_of_lists[key_id].keys():
            #     dict_of_lists[key_id]["acc_gen_normalized"] = []
            # dict_of_lists[key_id]["acc_gen_normalized"].append(all_results[key_seed][key_exp]["acc_gen_normalized"])

            if "loss_generalization" not in dict_of_lists[key_id].keys():
                dict_of_lists[key_id]["loss_generalization"] = []
            dict_of_lists[key_id]["loss_generalization"].append(all_results[key_seed][key_exp]["loss_generalization"])

            if "gradient_mean" not in dict_of_lists[key_id].keys():
                dict_of_lists[key_id]["gradient_mean"] = []
            dict_of_lists[key_id]["gradient_mean"].append(all_results[key_seed][key_exp]["gradient_mean"])

            if "gradient_mean_unormalized" not in dict_of_lists[key_id].keys():
                dict_of_lists[key_id]["gradient_mean_unormalized"] = []
            dict_of_lists[key_id]["gradient_mean_unormalized"].append(all_results[key_seed][key_exp]["gradient_mean_unormalized"])

    # turn the dict_of_lists into the desired dict
    for key_id in tqdm(dict_of_lists.keys()):

        acc_gen_list = np.array(dict_of_lists[key_id]["acc_generalization"])
        # acc_gen_norm_list = np.array(dict_of_lists[key_id]["acc_gen_normalized"])
        loss_gen_list = np.array(dict_of_lists[key_id]["loss_generalization"])
        gradient_list = np.array(dict_of_lists[key_id]["gradient_mean"])
        gradient_list_unormalized = np.array(dict_of_lists[key_id]["gradient_mean_unormalized"])

        assert len(acc_gen_list) == len(loss_gen_list),\
                         (len(acc_gen_list), len(loss_gen_list))
        # assert len(acc_gen_norm_list) == len(loss_gen_list),\
        #                  (len(acc_gen_norm_list), len(loss_gen_list))
        assert len(acc_gen_list) == len(gradient_list),\
                         (len(acc_gen_list), len(gradient_list))
        # assert len(acc_gen_list) == len(all_results.keys()),\
        #                  (len(acc_gen_list), len(all_results.keys()))

        n_seed = len(acc_gen_list)

        # mean
        dict_of_lists[key_id]["acc_generalization"] = acc_gen_list.mean()
        # dict_of_lists[key_id]["acc_gen_normalized"] = acc_gen_list.mean()
        dict_of_lists[key_id]["loss_generalization"] = loss_gen_list.mean()
        dict_of_lists[key_id]["gradient_mean"] = gradient_list.mean()
        dict_of_lists[key_id]["gradient_mean_unormalized"] = gradient_list_unormalized.mean()
        

        # standard deviation
        if n_seed >= 2:
            deviation = np.sqrt(((acc_gen_list - acc_gen_list.mean())**2).sum() / (n_seed - 1))
            dict_of_lists[key_id]["acc_generalization_deviation"] = deviation
            # deviation = np.sqrt(((acc_gen_norm_list - acc_gen_norm_list.mean())**2).sum() / (n_seed - 1))
            # dict_of_lists[key_id]["acc_gen_norm_deviation"] = deviation
            deviation = np.sqrt(((loss_gen_list - loss_gen_list.mean())**2).sum() / (n_seed - 1))
            dict_of_lists[key_id]["loss_generalization_deviation"] = deviation
            deviation = np.sqrt(((gradient_list - gradient_list.mean())**2).sum() / (n_seed - 1))
            dict_of_lists[key_id]["gradient_deviation"] = deviation
            deviation = np.sqrt(((gradient_list_unormalized - gradient_list_unormalized.mean())**2).sum() / (n_seed - 1))
            dict_of_lists[key_id]["gradient_deviation_unormalized"] = deviation
        else:
            deviation = None
            dict_of_lists[key_id]["acc_generalization_deviation"] = deviation
            # dict_of_lists[key_id]["acc_gen_norm_deviation"] = deviation
            dict_of_lists[key_id]["loss_generalization_deviation"] = deviation
            dict_of_lists[key_id]["gradient_deviation"] = deviation
            dict_of_lists[key_id]["gradient_deviation_unormalized"] = deviation
    
    if n_seed == 1:
            logger.warning("Only 1 seed found")

    return dict_of_lists

def average_from_json(json_path: str):

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    averaged = average_results(results)

    # all_results
    output_path = json_path.parent / f"average_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Collecting all results in {str(output_path)}")

    with open(str(output_path), "w") as output_file:
        json.dump(averaged, output_file, indent=2)


def merge_json(result_dir: str):

    result_dir = Path(result_dir)
    assert result_dir.is_dir(), str(result_dir)

    # We create one json file for each seed
    seed_dir_list = [seed for seed in result_dir.glob("*") if seed.is_dir() and "figure" not in seed.stem]
    logger.info(f"Found experiments for {len(seed_dir_list)} seeds")

    final_results = {}

    for seed in tqdm(seed_dir_list):

        # Create the resulting dict
        seed_results = {}

        json_list = [p for p in seed.rglob("*.json") if p.stem.startswith("result") and "seed" not in p.stem]
        n = 0 
        for p in json_list:
            with open(str(p), "r") as json_file:
                res = json.load(json_file)
                seed_results.update({str(n): res})
            n += 1
        final_results.update({str(seed.stem): seed_results})

        output_path = seed / f"results_seed_{seed.stem}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Collecting the results of {len(json_list)} experiments in {str(output_path)}")

        with open(str(output_path), "w") as output_file:
            json.dump(seed_results, output_file, indent=2)

    # all_results
    output_path = result_dir / f"all_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Collecting all results in {str(output_path)}")

    with open(str(output_path), "w") as output_file:
        json.dump(final_results, output_file, indent=2)

    # average_results
    average_dict = average_results(final_results)

    average_path = result_dir / f"average_results.json"
    average_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Collecting averaged results in {str(average_path)}")

    with open(str(average_path), "w") as average_file:
        json.dump(average_dict, average_file, indent=2)



def main(path_or_dir: str, mode: str = "merge"):

    if mode == "merge":
        merge_json(path_or_dir)
    elif mode == "average":
        average_from_json(path_or_dir)
    else:
        raise NotImplementedError()



if __name__ == "__main__":
    fire.Fire(main)

    


