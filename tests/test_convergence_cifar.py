import json
import tempfile
from pathlib import Path

from last_point.convergence_experiment import main as convergence_experiment

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
        "normalization",
        "gradient_mean",
        "K_constant",
        "n_params",
        "bias",
        "estimated_bound",
        "converged",
        'final_train_accuracy']


def test_simulation():

    alpha = 1.8

    with tempfile.TemporaryDirectory() as output_dir:

        train_accs, val_accs, _, result_dir = convergence_experiment(
            output_dir,
            3,
            3,
            n_ergodic=4,
            id_sigma=0,
            id_alpha=0,
            width=10,
            depth=1,
            subset=0.001,
            resize=10,
            eta=0.01,
            alpha=alpha,
            batch_size=4,
            data_type="cifar10",
            model="cnn",
        )

        assert len(train_accs) == len(val_accs)
        assert len(train_accs) == 4, (len(train_accs), 4)

        result_dir = Path(result_dir)
        assert result_dir.is_dir()

        json_path = result_dir / 'result_0_0.json'
        assert json_path.exists()
        with open(str(json_path), "r") as result_file:
            res = json.load(result_file)
            for key in KEYS:
                assert key in res.keys()
            sigma = res['sigma']
            assert res['alpha'] == 1.8

        fig_name = (f"loss_sigma_{sigma}_alpha_{alpha}").replace(".", "_")
        output_path = (result_dir / fig_name).with_suffix(".png")
        assert output_path.exists(), "No result file created"

        fig_name = (f"accuracies_sigma_{sigma}_alpha_{alpha}").replace(".", "_")
        output_path = (result_dir / fig_name).with_suffix(".png")
        assert output_path.exists(), "No result file created"








