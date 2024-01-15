import json
from pathlib import Path

import fire
from loguru import logger

def concatenate(json1: str, json2: str):
    """
    Concatenate two files in an all_resutls format
    """

    json1 = Path(json1)
    assert json1.exists(), str(json1)

    with open(str(json1), "r") as json_file:
        results_1 = json.load(json_file)

    json2 = Path(json2)
    assert json2.exists(), str(json2)

    with open(str(json2), "r") as json_file:
        results_2 = json.load(json_file)

    concatenated = {}

    for key in results_1.keys():
        new_key = "_".join([str(key), "a"])
        concatenated[new_key] = results_1[key]

        key_example = list(results_1.keys())[0]
        n_sigma_id = 1 + max([results_1[key_example][s]["id_sigma"] for s in results_1[key_example].keys()])
        # n_alpha_id = 1 + max([results_1[key_example][s]["id_alpha"] for s in results_1[key_example].keys()])

    
    logger.info(f"n_sigma_id: {n_sigma_id}")
    for key in results_2.keys():
        # Adapt the indices
        for a in results_2[key].keys():
            results_2[key][a]["id_sigma"] = results_2[key][a]["id_sigma"] + n_sigma_id
    
    for key in results_2.keys():
        new_key = "_".join([str(key), "b"])
        concatenated[new_key] = results_2[key]

    output_dir = json1.parent.parent / ("_".join([json1.parent.stem, json2.parent.stem]))
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / json1.name

    logger.info(f"Saving concatenated file in {str(output_path)}")
    logger.debug(f"Number of keys in the final dict: {len(concatenated.keys())}")
    with open(str(output_path), "w") as output_file:
        json.dump(concatenated, output_file, indent=2)

    
if __name__ == "__main__":
    fire.Fire(concatenate)
