import json
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from scipy.special import gamma
from tqdm import tqdm

from last_point.gaussian_mixture import sample_standard_gaussian_mixture
from last_point.simulation import run_one_simulation

"""
An experiment is characterized by:
 - A data distribution

The methods should be:
 - an empirical risk
 - a risk: potentially estimated from a validation set generated only one time
 - in that case, we use torch to estimate the gradients
 - the whole simulation is in one method
 - we only evaluate the validation set on the last point.
 - so no data_proxy is needed
 - everything is estimated: the squared case was stupid because of its multiplicativity properties
"""

class Simulation:

    def __init__(self, 
                 d: int,
                 n: int,
                 n_val: int = 100,
                 n_classes: int = 2,
                 sigma_min: float = 0.0001,
                 sigma_max: float = 1.,
                 n_sigma: int = 1,
                 alpha_min: float = 1.1,
                 alpha_max: float = 1.95,
                 n_alpha: int = 10,
                 w_init_std: float = 0.1,
                 normalization: bool = False,
                 seed: int = None,
                 momentum: float = 0.,
                 depth: int = 1,
                 width: int = 50
                 ):

        if seed is None:
            seed = np.random.randint(10000)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.seed: int = seed
        
        self.d: int = d
        self.sigma_min: float = sigma_min
        self.sigma_max: float =  sigma_max
        self.n_sigma: int = n_sigma
        self.alpha_min: float = alpha_min
        self.alpha_max: float = alpha_max
        self.n_alpha: int = n_alpha
        self.w_init_std: float = w_init_std
        self.n: int = n
        self.n_classes: int = n_classes
        self.n_val: int = n_val
        self.normalization: bool = normalization
        self.momentum: float = momentum
        self.depth: int = depth
        self.width: int = width

    def __str__(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @staticmethod
    def stable_normalization(alpha: float, d: float) -> float:

        alpha_factor = 2. * alpha * np.power(2., alpha - 1) / ((2. - alpha) * gamma(1. - alpha/2.))
        alpha_dim_factor = gamma((d + alpha) / 2.) / (d * gamma(d / 2.))

        return np.power(alpha_factor * alpha_dim_factor, 1. / alpha)

    @staticmethod
    def linear_regression(x_tab: np.ndarray, 
                          y_tab: np.ndarray, 
                          threshold: float = 1.e-6) -> float:
        """
        x_tab and y_tab are supposed to be one dimensional
        ie the data is scalar
        this performs linear regression y = ax + b and returns a
        """
        assert x_tab.ndim == 1, x_tab.shape
        assert y_tab.ndim == 1, y_tab.shape
        assert x_tab.shape == y_tab.shape, (x_tab.shape, y_tab.shape)

        n = len(x_tab)

        num = (x_tab * y_tab).sum() - x_tab.sum() * y_tab.sum() / n
        den = (x_tab * x_tab).sum() - x_tab.sum()**2 / n

        if den < threshold:
            logger.warning("Inifnite or undefined slope")
            return None
        
        return num / den


    def simulation(self, 
                   horizon:int,
                    n_ergodic: int = 100,
                    eta: float = 0.01):

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"on device {str(device)}")

        gen_grid = np.zeros((self.n_sigma, self.n_alpha))
        acc_gen_grid = np.zeros((self.n_sigma, self.n_alpha))

        # generate data
        n_per_class_train = self.n // self.n_classes
        x_train, y_train, means = sample_standard_gaussian_mixture(self.d, n_per_class_train)
        n_per_class_val = self.n_val // self.n_classes
        x_val, y_val, _ = sample_standard_gaussian_mixture(self.d, n_per_class_val, 
                                                           random_centers=False, means_deterministic=means)

        data = (x_train, y_train, x_val, y_val)

        # generate sigma and alpha tabs
        sigma_tab = np.exp(np.linspace(np.log(self.sigma_min),
                                        np.log(self.sigma_max),
                                          self.n_sigma))
        alpha_tab = np.linspace(self.alpha_min, self.alpha_max, self.n_alpha)

        # Generate initialization that will be shared among simulations
        initialization_noise = self.w_init_std * \
            torch.randn(size=(self.n_classes,self.d)).to(device)
        initialization = means.float().to(device) + initialization_noise

        # Initialize some logging
        losses = []
        accuracies = []
        estimators = []

        for s in tqdm(range(self.n_sigma)):
            for a in tqdm(range(self.n_alpha)):

                if self.normalization:
                    sigma_simu = Simulation.stable_normalization(alpha_tab[a], self.d) * sigma_tab[s]
                else:
                    sigma_simu = sigma_tab[s]

                generalization, loss_tab, \
                     accuracy_tab, estimator = run_one_simulation(horizon,
                                                       self.d,
                                                       eta,
                                                       sigma_simu,
                                                       alpha_tab[a],
                                                       initialization,
                                                       data,
                                                       n_ergodic,
                                                       n_classes=self.n_classes,
                                                       momentum=self.momentum,
                                                       width=self.width,
                                                       depth=self.depth)
                gen_grid[s, a] = generalization
                acc_gen_grid[s, a] = 100. * (accuracy_tab[-1][1] - accuracy_tab[-1][0])


                # For logging
                losses.append(loss_tab)
                accuracies.append(accuracy_tab)
                estimators.append(estimator)

                logger.info(f"train accuracy: {round(100. * accuracy_tab[-1][0].item(), 2)} %")
        
        logger.info(f"{self.n_sigma * self.n_alpha} simulations completed successfully")

        return acc_gen_grid, sigma_tab, alpha_tab, losses, accuracies, data, estimators, gen_grid


    def all_linear_regression(self,
                                gen_grid: np.ndarray,
                                sigma_tab: np.ndarray,
                                alpha_tab: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Returns the regression of the gen with respect to log(1/sigma), for each alpha
        and the regression of the gen with respect to alpha, for each sigma
        """

        # Regression gen/log(1/sigma)
        alpha_reg = np.zeros(self.n_alpha)
        for a in range(self.n_alpha):
            alpha_reg[a] = Simulation.linear_regression(np.log(1./sigma_tab), gen_grid[:, a])

        # Regression gen/alpha
        correlation_reg = np.zeros(self.n_sigma)
        for s in range(self.n_sigma):
            correlation_reg[s] = Simulation.linear_regression(alpha_tab, gen_grid[s, :])

        return alpha_reg, correlation_reg
    

    def plot_performance(self,
                         loss_tab,
                         accuracy_tab,
                         sigma_tab,
                         alpha_tab,
                         data,
                         estimators,
                         output_dir: str):
        """
        accuracy_tab[k] should be (train, validation)
        """
        
        assert len(loss_tab) == len(accuracy_tab),\
              (len(loss_tab), len(accuracy_tab))
        assert len(loss_tab[0]) == len(accuracy_tab[0])
        iterations = len(loss_tab[0])

        if not Path(output_dir).is_dir():
            Path(output_dir).mkdir()

        output_dir = Path(output_dir) / f"results_d_{self.d}_{int(time.time())}"
        if not output_dir.is_dir():
            output_dir.mkdir()

        json_path = (output_dir / "simulation").with_suffix(".json")
        logger.info(f"Saving JSON file in {str(json_path)}")
        with open(str(json_path), "w") as json_file:
            json.dump(self.__dict__, json_file, indent = 2)

        logger.info(f"Saving all figures in {str(output_dir)}") 
        k = 0 # TODO: this is a hack, find a better solution
        for s in tqdm(range(self.n_sigma)):
            for a in tqdm(range(self.n_alpha)):

                train_accs = [accuracy_tab[k][i][0] for i in range(iterations)]
                val_accs = [accuracy_tab[k][i][1] for i in range(iterations)]

                train_losses = [loss_tab[k][i][0] for i in range(iterations)]
                val_losses = [loss_tab[k][i][1] for i in range(iterations)]

                # evolution  of the training and validation losses
                plt.figure()
                plt.plot(np.arange(iterations), train_losses, label="Train BCE")
                plt.plot(np.arange(iterations), val_losses, label="Validation BCE")
                fig_name = (f"loss_sigma_{sigma_tab[s]}_alpha_{alpha_tab[a]}").replace(".","_")
                output_path = (output_dir / fig_name).with_suffix(".png")
                plt.legend()
                plt.savefig(str(output_path))
                plt.close()      

                # evolution of both accuracies
                plt.figure()
                plt.plot(np.arange(iterations), train_accs, label="Train accuracy")
                plt.plot(np.arange(iterations), val_accs, label="Test accuracy")
                fig_name = (f"accuracies_sigma_{sigma_tab[s]}_alpha_{alpha_tab[a]}").replace(".","_")
                output_path = (output_dir / fig_name).with_suffix(".png")
                plt.legend()
                plt.savefig(str(output_path))
                plt.close()

                if self.d == 2:
                    # predicted labels in the plane
                    plt.figure()
                    predictions_train = np.argmax(estimators[k][0], axis=1)
                    predicted_0 = data[0][(predictions_train==0)]
                    plt.scatter(predicted_0[:, 0], predicted_0[:, 1], label="predicted 0 train")
                    predicted_1 = data[0][(predictions_train==1)]
                    plt.scatter(predicted_1[:, 0], predicted_1[:, 1], label="predicted 1 train")
                    fig_name = (f"predictions_sigma_{sigma_tab[s]}_alpha_{alpha_tab[a]}").replace(".","_")
                    output_path = (output_dir / fig_name).with_suffix(".png")
                    plt.legend()
                    plt.savefig(str(output_path))
                    plt.close()

                    # ground truth in the plane
                    plt.figure()
                    ground_truth = data[1]
                    predicted_0 = data[0][(ground_truth==0)]
                    plt.scatter(predicted_0[:, 0], predicted_0[:, 1], label="predicted 0 train")
                    predicted_1 = data[0][(ground_truth==1)]
                    plt.scatter(predicted_1[:, 0], predicted_1[:, 1], label="predicted 1 train")
                    fig_name = (f"ground_truth_sigma_{sigma_tab[s]}_alpha_{alpha_tab[a]}").replace(".","_")
                    output_path = (output_dir / fig_name).with_suffix(".png")
                    plt.legend()
                    plt.savefig(str(output_path))
                    plt.close()

                    # same things for the validation set
                    plt.figure()
                    predictions_train = np.argmax(estimators[k][1], axis=1)
                    predicted_0 = data[2][(predictions_train==0)]
                    plt.scatter(predicted_0[:, 0], predicted_0[:, 1], label="predicted 0 train")
                    predicted_1 = data[2][(predictions_train==1)]
                    plt.scatter(predicted_1[:, 0], predicted_1[:, 1], label="predicted 1 train")
                    fig_name = (f"predictions_val_sigma_{sigma_tab[s]}_alpha_{alpha_tab[a]}").replace(".","_")
                    output_path = (output_dir / fig_name).with_suffix(".png")
                    plt.legend()
                    plt.savefig(str(output_path))
                    plt.close()

                    plt.figure()
                    ground_truth_val = data[3]
                    predicted_0 = data[2][(ground_truth_val==0)]
                    plt.scatter(predicted_0[:, 0], predicted_0[:, 1], label="predicted 0 train")
                    predicted_1 = data[2][(ground_truth_val==1)]
                    plt.scatter(predicted_1[:, 0], predicted_1[:, 1], label="predicted 1 train")
                    fig_name = (f"ground_truth_val_sigma_{sigma_tab[s]}_alpha_{alpha_tab[a]}").replace(".","_")
                    output_path = (output_dir / fig_name).with_suffix(".png")
                    plt.legend()
                    plt.savefig(str(output_path))
                    plt.close()

                k += 1
        
    
    def plot_results(self,
                      gen_grid: np.ndarray,
                      sigma_tab: np.ndarray,
                      alpha_tab: np.ndarray,
                      output_dir: str,
                      horizon: int):
        
        if not Path(output_dir).is_dir():
            Path(output_dir).mkdir()

        output_dir = Path(output_dir) / f"results_d_{self.d}_{int(time.time())}"
        if not output_dir.is_dir():
            output_dir.mkdir()

        json_path = (output_dir / "simulation").with_suffix(".json")
        logger.info(f"Saving JSON file in {str(json_path)}")
        with open(str(json_path), "w") as json_file:
            saved_dict = self.__dict__
            saved_dict["horizon"] = horizon
            json.dump(saved_dict, json_file, indent = 2)

        npy_path = (output_dir / "generalization").with_suffix(".npy")
        logger.info(f"Saving NPY file in {str(npy_path)}")
        np.save(str(npy_path), gen_grid)

        logger.info(f"Saving all figures in {str(output_dir)}") 
        for s in tqdm(range(self.n_sigma)):

            plt.figure()
            plt.scatter(alpha_tab, gen_grid[s, :])          
            plt.title(f'Generalization for sigma = {sigma_tab[s]}')

            # Saving the figure
            fig_name = (f"sigma_{sigma_tab[s]}").replace(".","_")
            output_path = (output_dir / fig_name).with_suffix(".png")
            plt.savefig(str(output_path))
            plt.close()

        for a in tqdm(range(self.n_alpha)):

            plt.figure()
            plt.scatter(sigma_tab, gen_grid[:, a])
            plt.xscale("log")          
            plt.title(f'Generalization for alpha = {alpha_tab[a]}')

            # Saving the figure
            fig_name = (f"alpha_{alpha_tab[a]}").replace(".","_")
            output_path = (output_dir / fig_name).with_suffix(".png")
            plt.savefig(str(output_path))
            plt.close()


        # Finally: the linear regressions
        alpha_reg, correlation_reg = self.all_linear_regression(
            gen_grid,
            sigma_tab,
            alpha_tab
        )
        
        if all(alpha_reg[k] is not None for k in range(self.n_alpha)):
            alpha_reg_path = (output_dir / "alpha_regression").with_suffix(".png")
            plt.figure()
            plt.plot(np.linspace(1,2, 100), np.linspace(1,2,100), color = "r")
            plt.scatter(alpha_tab, alpha_reg)
            plt.title("Regression of alpha from the generalization bound")
            plt.savefig(str(alpha_reg_path))
            plt.close()


        if all(correlation_reg[k] is not None for k in range(self.n_sigma)):
            correlation_reg_path = (output_dir / "correlation_regression").with_suffix(".png")
            plt.figure()
            plt.scatter(sigma_tab, correlation_reg)
            plt.yscale("log")
            plt.title("Correlation between generalization and alpha, in function of sigma")
            plt.savefig(str(correlation_reg_path))
            plt.close()



        
def main(n=1000,
          d = 10, 
          n_val = 1000,
          eta=0.001,\
          horizon=20000,
          n_ergodic=1000,
          n_sigma: int=10,
          n_alpha: int = 10, 
          init_std: float = 1.,
          normalization: bool = False,
          sigma_min = 0.001,        
          sigma_max = 0.1,
          output_dir: str = "figures",
          depth: int = 1,
          width: int = 50):

    simulator = Simulation(d, n, n_sigma=n_sigma, n_alpha=n_alpha,\
                           w_init_std=init_std, n_val=n_val,
                             normalization=normalization, sigma_min=sigma_min,
                             sigma_max=sigma_max, depth=depth,
                             width=width)

    gen_grid, sigma_tab, alpha_tab, *_ = simulator.simulation(horizon,
                                                          n_ergodic,
                                                          eta)
    
    simulator.plot_results(gen_grid, sigma_tab, alpha_tab, output_dir, horizon)


if __name__ == "__main__":
    """
    Test command: 
    PYTHONPATH=$PWD python -m pdb last_point/experiments.py --n 10 --d 2 --n_val 10 --horizon 10 --n_sigma 3 --n_alpha 3 --output_dir tests --depth 1 --width 2
    """

    fire.Fire(main)