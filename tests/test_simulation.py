import json
import tempfile
from pathlib import Path

from last_point.simulation import run_and_save_one_simulation

KEYS = [
    "horizon",
        "input_dimension",
        "n",
        "n_val",
        "data_seed",
        "model_seed",
        "eta",
        "sigma",
        "alpha",
        "n_ergodic",
        "n_classes",
        "decay",
        "depth",
        "width",
        "loss_generalization",
        "acc_generalization",
        "id_sigma",
        "id_alpha",
        "normalization_factor",
        "normalization",
        "gradient_mean",
        "K_constant",
        "n_params",
        "bias",
        "estimated_bound",
        "converged",
        'final_train_accuracy',
        "acc_gen_normalized",
        "gradient_mean_unormalized",
        "resize",
        "scale_sigma",
        "normalization",
        "stopping",
        "batch_size",
        "id_eta"
]


def test_simulation():

    with tempfile.TemporaryDirectory() as output_dir:

        run_and_save_one_simulation(
            output_dir,
            3,
            n_ergodic=4,
            id_sigma=0,
            id_alpha=0,
            id_eta=0,
            width=10,
            depth=1,
            subset=0.001,
            resize=10,
            eta=0.01
        )

        output_dir = Path(output_dir)

        eta_name = str(0.01).replace(".","_")
        rname = f"result_{0}_{0}_{10}_{eta_name}_{-1}"
        result_path = (output_dir / rname).with_suffix(".json")

        assert result_path.exists(), "No result file created"

        with open(str(result_path), "r") as result_file:
            res = json.load(result_file)
            for key in KEYS:
                assert key in res.keys()






