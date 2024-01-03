import json
from pathlib import Path

import fire
import numpy as np
from loguru import logger
from tqdm import tqdm

from last_point.utils import poly_alpha

def main(json_path: str):

    json_path = Path(json_path)
    assert json_path.exists(), str(json_path)

    with open(str(json_path), "r") as json_file:
        results = json.load(json_file)

    for key in tqdm(results.keys()):

        gen = results[key]["acc_generalization"]
        constant = poly_alpha(results[key]["alpha"])

        normalized_gen = gen / np.sqrt(constant)
        results[key]["acc_gen_normalized"] = normalized_gen

    with open(str(json_path), "w") as output_file:
        json.dump(results, output_file, indent=2)


if __name__ == "__main__":
    fire.Fire(main)

    



    

    

