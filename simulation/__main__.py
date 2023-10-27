import fire

from simulation.gaussian_simulation import GaussianExperiment


def main(n: int, d: int, horizon: int = 100, n_exp: int = 20,\
          eta: float = 0.001, sigma: float = 0.01):

    experiment = GaussianExperiment(horizon, n, d, eta, sigma)
    alpha_tab, gen_T_tab, gen_sup_tab = experiment.run_simulations(n_exp=n_exp, n_dataset=20,\
                                                                    n_alpha=10, alpha_min=1.2)
    experiment.plot_results_one_simulation(alpha_tab,\
                                            gen_T_tab.mean(axis=0),\
                                                  gen_sup_tab.mean(axis=0))


if __name__ == "__main__":
    fire.Fire(main)
