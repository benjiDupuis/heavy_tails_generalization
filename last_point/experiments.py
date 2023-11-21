import json
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
from typing import Tuple

from simulation.levy import generate_levy_for_simulation

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

class LinearModel(nn.Module):

    def __init__(self, 
                 input_dim: int = 10,
                 bias: bool = False,
                 n_classes: int = 2):
        super(LinearModel, self).__init__()
        
        self.input_dim: int = input_dim
        self.bias: bool = bias
        self.layer = nn.Linear(self.input_dim, n_classes, bias = self.bias)

    def get_vector(self):
        # TODO: implement this
        pass

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim) # I don't know why we have this
        x = self.layer(x)
        return x
    
    @torch.no_grad()
    def add_noise(self, noise: torch.Tensor):
        # This methods add a noise to the parameters of the networks
        assert not self.bias, "For now we don't handle biases" 
        # self.layer.weight.data = self.layer.weight.data + noise
        self.layer.weight.add_(noise)

    @torch.no_grad()
    def initialization(self, w: torch.Tensor):
        assert not self.bias, "For now we don't handle biases" 
        assert self.layer.weight.data.shape == w.shape,\
              (self.layer.weight.data.shape, w.shape)
        self.layer.weight.data = w



@torch.no_grad()
def sample_standard_gaussian_mixture(dimension: int,
                        n_per_class: int,
                        n_classes: int = 2,
                        means_std: float = 25,
                        blobs_std: float = 100.)-> (torch.Tensor, torch.Tensor):
    
    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Generate means of the blobs
    means = means_std * np.random.normal(0., 1., size = n_classes)

    # Generate each blob
    # WARNING: it only works with scale standard gaussians 
    blobs = []
    labels = []
    for i in range(n_classes):
        blobs.append(means[i] +\
                      blobs_std * torch.randn(size=(n_per_class, dimension)))    
        labels.append(i * torch.ones(n_per_class, dtype=torch.int64))

    # concatenate and random shuffle
    x = torch.concatenate(blobs)
    assert x.shape == (n_per_class * n_classes, dimension), x.shape
    y = torch.concatenate(labels)
    assert y.ndim == 1
    assert y.shape[0] == n_per_class * n_classes, y.shape[0]

    indices = list(np.arange(n_per_class * n_classes))
    np.random.shuffle(indices)

    return x.to(device)[indices, ...], y.to(device)[indices]
    


def run_one_simulation(horizon: int, 
                        d: int,
                        eta: float,
                        sigma: float,
                        alpha: float,
                        initialization: torch.Tensor,
                        data: Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor],
                        n_ergodic: int = 100,
                        n_classes: int = 2):
    """
    Data format should be (x_train, y_train, x_val, y_val)
    """
    
    # Sanity checks
    assert horizon > 2, horizon
    
    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Seed
    torch.random.seed()

    # Define data
    assert len(data) == 4, len(data)
    x_train = data[0].to(device)
    y_train = data[1].to(device)
    x_val = data[2].to(device)
    y_val = data[3].to(device)
    assert x_train.ndim == 2
    assert y_train.ndim == 1
    assert x_val.ndim == 2
    assert y_val.ndim == 1
    assert x_train.shape[0] == y_train.shape[0]
    assert x_train.shape[1] == x_val.shape[1]

    n = x_train.shape[0]

    # Define model, loss and optimizer
    model = LinearModel(d, n_classes = n_classes).to(device) 
    with torch.no_grad():
        model.initialization(initialization)
    opt = torch.optim.SGD(model.parameters(), lr = eta)
    crit = nn.CrossEntropyLoss().to(device)

    loss_tab = []
    gen_tab = []

    # Generate all noise
    n_params = d * n_classes
    noise = sigma * generate_levy_for_simulation(n_params, \
                                         horizon + n_ergodic,
                                         alpha,
                                         eta)
    # Loop
    for k in range(horizon + n_ergodic):

        # Validation if we are after the time horizon
        if k >= horizon:
            with torch.no_grad():
                out_val = model(x_val)
                loss_val = crit(out_val, y_val)

        # evaluation of the empirical loss
        # keep in mind that this is a full batch experiment
        opt.zero_grad()
        out = model(x_train)

        assert out.shape == (n, n_classes), out.shape
        loss = crit(out, y_train)

        if torch.isnan(loss):
            logger.error('Loss has gone nan ❌')
            break

        # calculate the gradients
        loss.backward()

        # Logging
        loss_tab.append(loss.item())
        if k >= horizon:
            gen_tab.append(loss_val.item() - loss.item())

        # Gradient step
        opt.step()

        # Adding the levy noise
        with torch.no_grad():
            model.add_noise(torch.from_numpy(noise[k].reshape(n_classes, d)).to(device))

        

    # Compute the estimated generalization at the end
    gen_tab = np.array(gen_tab)
    generalization = gen_tab.mean()

    return generalization, loss_tab




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
                 ):
        
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

    def simulation(self, 
                   horizon:int,
                    n_ergodic: int = 100,
                    eta: float = 0.01):

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info(f"on device {str(device)}")

        gen_grid = np.zeros((self.n_sigma, self.n_alpha))

        # generate data
        n_per_class_train = self.n // self.n_classes
        x_train, y_train = sample_standard_gaussian_mixture(self.d, n_per_class_train)
        n_per_class_val = self.n_val // self.n_classes
        x_val, y_val = sample_standard_gaussian_mixture(self.d, n_per_class_val)

        data = (x_train, y_train, x_val, y_val)

        # generate sigma and alpha tabs
        sigma_tab = np.exp(np.linspace(np.log(self.sigma_min),
                                        np.log(self.sigma_max),
                                          self.n_sigma))
        alpha_tab = np.linspace(self.alpha_min, self.alpha_max, self.n_alpha)

        # Generate initialization that will be shared among simulations
        initialization = self.w_init_std * \
            torch.randn(size=(self.n_classes,self.d)).to(device)

        for s in tqdm(range(self.n_sigma)):
            for a in tqdm(range(self.n_alpha)):

                generalization, _ = run_one_simulation(horizon,
                                                       self.d,
                                                       eta,
                                                       sigma_tab[s],
                                                       alpha_tab[a],
                                                       initialization,
                                                       data,
                                                       n_ergodic,
                                                       n_classes=self.n_classes)
                gen_grid[s, a] = generalization
        
        logger.info(f"{self.n_sigma * self.n_alpha} simulations completed successfully")

        return gen_grid, sigma_tab, alpha_tab
    
    def plot_results(self,
                      gen_grid: np.ndarray,
                      sigma_tab: np.ndarray,
                      alpha_tab: np.ndarray,
                      output_dir: str):
        
        if not Path(output_dir).is_dir():
            Path(output_dir).mkdir()

        output_dir = Path(output_dir) / f"results_d_{self.d}_{int(time.time())}"
        if not output_dir.is_dir():
            output_dir.mkdir()

        logger.info(f"Saving all results in {str(output_dir)}")

        for s in tqdm(range(self.n_sigma)):

            plt.figure()

            plt.scatter(alpha_tab, 
                     gen_grid[s, :])
            
            plt.title(f'Generalization for sigma = {sigma_tab[s]}')

            # Saving the figure
            fig_name = (f"sigma_{sigma_tab[s]}").replace(".","_")
            output_path = (output_dir / fig_name).with_suffix(".png")
            plt.savefig(str(output_path))

        for a in tqdm(range(self.n_alpha)):

            plt.figure()

            plt.scatter(sigma_tab, 
                     gen_grid[:, a])
            plt.xscale("log")
            
            plt.title(f'Generalization for alpha = {alpha_tab[a]}')

            # Saving the figure
            fig_name = (f"alpha_{alpha_tab[a]}").replace(".","_")
            output_path = (output_dir / fig_name).with_suffix(".png")
            plt.savefig(str(output_path))

        
def main(n=100, d = 10, n_val = 100, eta=0.01,\
          horizon=1000, n_ergodic=20, n_sigma: int=5,
          n_alpha: int = 5, init_std: float = 1.):

    simulator = Simulation(d, n, n_sigma=n_sigma, n_alpha=n_alpha,\
                           w_init_std=init_std, n_val=n_val)

    gen_grid, sigma_tab, alpha_tab = simulator.simulation(horizon,
                                                          n_ergodic,
                                                          eta)
    
    simulator.plot_results(gen_grid, sigma_tab, alpha_tab, "figures")


if __name__ == "__main__":

    fire.Fire(main)