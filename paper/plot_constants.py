from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from last_point.utils import poly_alpha

def plot_F(output_dir: str = "paper"):

    alphas = np.linspace(1., 1.999, 100)
    F = poly_alpha(alphas)

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "F_factor.png"

    plt.figure()
    plt.plot(alphas, F, color="r")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$F(\alpha)$")
    # plt.title(r"Factor $F(\alpha)$ with respect to the tail-index $\alpha$")

    logger.info(f"saving figure in {str(output_path)}")
    plt.savefig(str(output_path))
    plt.close()

def plot_dimension_dependence(output_dir: str = "paper"):

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dimension_dependence.png"

    alphas = np.linspace(1., 2., 100)

    plt.figure()
    plt.xlim(1., 2.)
    plt.ylim(0., 1.5)

    plt.plot(alphas, np.ones(100), "--", color="gray")
    plt.plot(alphas, 0.5 * np.ones(100), "--", color="red", label="max. value for convergence in overparameterized regime")
    plt.plot(alphas, 1.5 * np.ones(100), "--", color="gray")

    plt.plot(alphas, 0.5 + alphas / 2., color = "k", label="Raj et al.")
    plt.plot(alphas, 1. - alphas / 2., color = "g", label="Ours")
    plt.plot(alphas, 2. - alphas / 2., "--", color = "g", label=r"Ours if $C\propto \sqrt{d}$")

    plt.plot(2. * np.ones(100), np.linspace(0., 1., 100), color="b", label="Known limit for Langevin dynamics")


    plt.legend()

    logger.info(f"saving figure in {str(output_path)}")
    plt.savefig(str(output_path))
    plt.close()


if __name__ == "__main__":
    fire.Fire(plot_dimension_dependence)

