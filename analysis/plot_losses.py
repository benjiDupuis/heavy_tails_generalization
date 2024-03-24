from pathlib import Path

import fire
import numpy as np
from loguru import logger
import matplotlib.pyplot as plt


####################
#This script is the one we used to plot the evolution of the test and train accuracies
###################


def plot_losses(result_dir: str):

    result_dir = Path(result_dir)

    train_path = result_dir / "train.npy"
    val_path = result_dir / "val.npy"
    alpha_path = result_dir / "alpha.npy"

    acc_train_tabs = np.load(str(train_path))
    acc_val_tabs = np.load(str(val_path))
    alpha_list = np.load(str(alpha_path))

    iterations = len(acc_train_tabs[0])
    
    plt.figure(figsize=(12,3))

    plt.subplot(141)
    plt.plot(np.arange(iterations), acc_train_tabs[0], label="Train accuracy")
    plt.plot(np.arange(iterations), acc_val_tabs[0], label="Test accuracy")
    plt.legend()
    plt.title(r"$\alpha=$" + f"{alpha_list[0]}")
    plt.legend()

    plt.subplot(142)
    plt.plot(np.arange(iterations), acc_train_tabs[1], label="Train accuracy")
    plt.plot(np.arange(iterations), acc_val_tabs[1], label="Test accuracy")
    plt.legend()
    plt.title(r"$\alpha=$" + f"{alpha_list[1]}")
    plt.legend()

    plt.subplot(143)
    plt.plot(np.arange(iterations), acc_train_tabs[2], label="Train accuracy")
    plt.plot(np.arange(iterations), acc_val_tabs[2], label="Test accuracy")
    plt.legend()
    plt.title(r"$\alpha=$" + f"{alpha_list[2]}")
    plt.legend()

    plt.subplot(144)
    plt.plot(np.arange(iterations), acc_train_tabs[3], label="Train accuracy")
    plt.plot(np.arange(iterations), acc_val_tabs[3], label="Test accuracy")
    plt.legend()
    plt.title(r"$\alpha=$" + f"{alpha_list[3]}")

    output_path = result_dir / "accuracies.png"
    plt.legend()
    logger.info(f'Saving accuracy plot in {str(output_path)}')
    plt.savefig(str(output_path))

    plt.close() 


if __name__ == "__main__":
    fire.Fire(plot_losses)
