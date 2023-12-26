import fire

from last_point.experiments import Simulation

def main(n=1000,
          d = 2, 
          n_val = 1000,
          eta = 0.001,\
          horizon = 20000,
          n_ergodic = 2,
          n_sigma: int = 1,
          n_alpha: int = 1, 
          init_std: float = 1.,
          normalization: bool = False,
          sigma_min = 0.1,        
          sigma_max = 1.,
          decay = 0.,
          alpha_min = 1.8,
          alpha_max = 2.,
          depth: int = 0,
          width: int = 50):

    simulator = Simulation(d, n, n_sigma=n_sigma, n_alpha=n_alpha,\
                           w_init_std=init_std, n_val=n_val,
                             normalization=normalization, sigma_min=sigma_min,
                             sigma_max=sigma_max, decay=decay,
                             alpha_min=alpha_min, alpha_max=alpha_max,
                             width=width, depth=depth)

    _, sigma_tab, alpha_tab, \
         loss_tab, accuracies, data, \
           estimators, _ = simulator.simulation(horizon,
                                                n_ergodic,
                                                eta)
    simulator.plot_performance(loss_tab, accuracies, 
                               sigma_tab, alpha_tab, 
                               data, estimators, "tests")



    

if __name__ == "__main__":
    fire.Fire(main)