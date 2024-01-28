from pathlib import Path

import fire
import numpy as np
from loguru import logger

from analysis.merge_json import merge_json
from last_point.simulation import run_and_save_one_simulation

RESULT_DIR = "results"

def main(result_dir: str=RESULT_DIR,
            horizon: int=10000, 
            d: int=2,
            eta: float=0.01,
            sigma_min: float= 0.5,
            sigma_max: float= 30.,
            alpha_min: float= 1.6,
            alpha_max: float= 2.,
            width_min: int= 40,
            width_max: int= 300,
            n_sigma: int= 1,
            n_alpha: int= 2,
            n_width: int= 2,
            n: int = 1000,
            n_val: int = 1000,
            n_ergodic: int = 100,
            n_classes: int = 2,
            decay: float = 0.001,
            depth: int = 1,
            normalization: bool = True,
            compute_gradient: bool = True,
            bias: bool = False,
            data_type: str = "mnist",
            subset: float = 0.1,
            resize: int = 28,
            classes: list = None,
            stopping: bool = False,
            scale_sigma: bool = True,
            batch_size: int = -1,
            n_seed: int = 1):

    alphas = np.linspace(alpha_min, alpha_max, n_alpha)
    sigmas = np.exp(np.linspace(np.log(sigma_min), np.log(sigma_max), n_sigma))
    widths = np.linspace(width_min, width_max, n_width, dtype=np.int64)

    logger.info(f"Launching {n_width * n_alpha * n_sigma} experiments.")

    for k in range(n_seed):

        np.random.seed(k)

        data_seed = np.random.randint(10000)
        model_seed = np.random.randint(10000)

        for a in range(n_alpha):
            for s in range(n_sigma):
                for w in range(n_width):

                    seed_results = str(Path(result_dir) / str(k))

                    run_and_save_one_simulation(
                        seed_results,
                        horizon, 
                        d,
                        eta,
                        sigmas[s],
                        alphas[a],
                        n,
                        n_val,
                        n_ergodic,
                        n_classes,
                        decay,
                        depth,
                        int(widths[w]),
                        data_seed,
                        model_seed,
                        normalization,
                        int(s),
                        int(a),
                        compute_gradient,
                        bias,
                        data_type,
                        subset,
                        resize,
                        classes,
                        stopping,
                        scale_sigma,
                        batch_size,
                        0
                    )

    merge_json(result_dir)


if __name__ == "__main__":
    fire.Fire(main)



    