import fire

import matplotlib.pyplot as plt
from loguru import logger

from last_point.experiments import Simulation

def main(n=1000,
          d = 2, 
          n_val = 1000,
          eta = 0.001,\
          horizon = 1000,
          n_ergodic = 100,
          n_sigma: int = 5,
          n_alpha: int = 5, 
          init_std: float = 1.,
          normalization: bool = False,
          sigma_min = 0.0001,        
          sigma_max = 1.):

    simulator = Simulation(d, n, n_sigma=n_sigma, n_alpha=n_alpha,\
                           w_init_std=init_std, n_val=n_val,
                             normalization=normalization, sigma_min=sigma_min,
                             sigma_max=sigma_max)

    _, sigma_tab, alpha_tab, \
         loss_tab, accuracies, data, \
           estimators = simulator.simulation(horizon,
                                                n_ergodic,
                                                eta)
    simulator.plot_performance(loss_tab, accuracies, 
                               sigma_tab, alpha_tab, 
                               data, estimators, "tests")



    

if __name__ == "__main__":
    fire.Fire(main)