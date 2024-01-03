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
    plt.title(r"Factor $F(\alpha)$ with respect to the tail-index $\alpha$")

    logger.info(f"saving figure in {str(output_path)}")
    plt.savefig(str(output_path))


if __name__ == "__main__":
    fire.Fire(plot_F)

