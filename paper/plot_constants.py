from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.special import gamma

from last_point.utils import poly_alpha

def test_constant(alpha):
    if type(alpha) == float and alpha == 2.:
        # asymptotic development of gamma(1-s)
        # using Euler reflection formula
        # TODO: recheck this
        num_alpha = 2.
    else:
        num_alpha = (2. - alpha) * gamma(1. - alpha / 2.)

    # den_alpha = alpha * np.power(2., alpha / 2.)
    den_alpha = alpha

    # return num_alpha / den_alpha
    return (2. - alpha) / np.sin(np.pi * 0.5 * alpha)

def plot_F(output_dir: str = "paper"):

    alphas = np.linspace(1., 1.999, 100)
    F = poly_alpha(alphas) #To obtain the true figure

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "F_factor.png"

    plt.figure()
    plt.plot(alphas, F, color="r", label=r"$\mathbf{\alpha \longmapsto P_\alpha}$")
    plt.xlabel(r"$\mathbf{\alpha}$", weight="bold")
    plt.ylabel(r"$\mathbf{P_\alpha}$", weight="bold")
    # plt.title(r"Factor $F(\alpha)$ with respect to the tail-index $\alpha$")

    logger.info(f"saving figure in {str(output_path)}")
    plt.legend()
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

    plt.fill_between(alphas, np.zeros(100), np.ones(100), color="g", alpha=0.2)
    plt.fill_between(alphas, np.ones(100), 1.5 * np.ones(100), color="r", alpha=0.2)

    plt.plot(alphas, 0.5 * np.ones(100), "--", color="gray")
    plt.plot(alphas, np.ones(100), "--", color="red", label="max. value for convergence in overparameterized regime")
    plt.plot(alphas, 1.5 * np.ones(100), "--", color="gray")

    plt.plot(alphas, 0.5 + alphas / 2., color = "k", label="Raj et al.")
    plt.plot(alphas, 1. - alphas / 2., color = "g", label="Ours")
    # plt.plot(alphas, 2. - alphas / 2., "--", color = "g", label=r"Ours if $C\propto \sqrt{d}$")

    # plt.plot(2. * np.ones(100), np.linspace(0., 1., 100), color="b",\
    #  label="Known limit for Langevin dynamics", linewidth=5.)

    plt.xlabel(r"$\mathbf{\alpha}$")
    plt.ylabel(r"$\mathbf{\frac{\text{rate in  }d}{\text{rate in  }n}}$")

    plt.legend()

    logger.info(f"saving figure in {str(output_path)}")
    plt.savefig(str(output_path))
    plt.close()


if __name__ == "__main__":
    fire.Fire(plot_dimension_dependence)

