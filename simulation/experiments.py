import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from tqdm import tqdm

from simulation.levy import generate_levy_for_simulation

"""
An experiment is characterized by:
 - A data distribution

The methods should be:
 - an empirical risk
 - a risk: potentially estimated from a validation set generated only one time
 - a gradient of the empirical risk
 - a gradient of the risk, also potentially estimated from a validation set
"""


class Experiment:

    def __init__(self,\
                  horizon: int, 
                  n: int,
                  d: int,
                  eta: float,
                  sigma: float = 1.,
                  seed: int = 56, 
                  starting_point_mean: float = 0.,
                  starting_point_std: float = 1.,
                  n_val: int = 1000,
                  save_dir: str = "figures"):
        """
        TODO: take into account surrogate losses
        """
        
        self.horizon: int = horizon
        self.n: float = n
        self.d: float = d
        self.eta: float = eta
        self.sigma: float = sigma
        self.starting_point_mean: float = starting_point_mean
        self.starting_point_std: float = starting_point_std

        data_val = self.generate_val_set(n_val)
        self.data_val: np.ndarray = data_val

        # To compute the gradient of the risk
        if self.data_val is None:
            data_val_proxy = None
        else:
            data_val_proxy = self.generate_data_proxy(data_val)
        self.data_val_proxy = data_val_proxy

        self.save_dir: Path = Path(save_dir)
        if not self.save_dir.is_dir():
            self.save_dir.mkdir()

        np.random.seed(seed)

    def experiment_type(self) -> str:
        return "experiment"

    def generate_levy(self, alpha: float, loc: float = 0.):
        return generate_levy_for_simulation(self.d, self.horizon,\
                                             alpha, self.eta, loc)

    def sample_data(self, m: int) -> np.ndarray:
        pass

    def generate_val_set(self, m: int) -> np.ndarray:
        return None
    
    def loss(self, w: np.ndarray, data: np.ndarray) -> float:
        pass

    def gradient_loss(self):
        pass

    def risk(self, w: np.ndarray) -> float:
        pass

    def estimate_risk(self, w: np.ndarray) -> float:
        pass

    def empirical_risk(self, w: np.ndarray, data: np.ndarray) -> float:
        pass

    def gradient_risk(self, w: np.ndarray) -> np.ndarray:
        pass

    def estimate_gradient_risk(self, w: np.ndarray) -> np.ndarray:
        pass

    def gradient_empirical_risk(self, w: np.ndarray,\
                                 data_proxy: np.ndarray) -> np.ndarray:
        pass

    def generate_data(self) -> np.ndarray:
        pass

    def generate_data_proxy(self, data: np.ndarray) -> np.ndarray:
        pass

    def initialization(self) -> np.ndarray:
        return np.random.normal(self.starting_point_mean,\
                                    self.starting_point_std,\
                                    size=self.d)

    def run_one_alpha_simulation(self, 
                                alpha: float,
                                w0: np.ndarray = None,
                                data: np.ndarray = None,
                                data_proxy: np.ndarray = None) \
                                    -> (np.ndarray, np.ndarray):

        if data is None:
            logger.warning(f"No data input for alpha = {alpha}, generating it.")
            data = self.generate_data()
            data_proxy = self.generate_data_proxy(data)
        assert data.shape == (self.n, self.d)
        assert data_proxy.shape == (self.d, self.d)

        risk_tab = np.zeros(self.horizon + 1)
        er_tab = np.zeros(self.horizon + 1)

        # Starting_point
        if w0 is None:
            logger.warning("No initial point given")
            w = self.initialization()
        else:
            w = w0.copy()

        risk_tab[0] = self.risk(w)
        er_tab[0] = self.empirical_risk(w, data)

        # logger.debug(risk_tab[0])

        # Generate all the levy increments
        levy = self.generate_levy(alpha, loc = 0.)
        assert levy.shape == (self.horizon, self.d)

        for t in range(1, self.horizon + 1):

            w = w - self.eta * self.gradient_empirical_risk(w, data_proxy) +\
                  self.sigma * levy[t-1, :]

            risk_tab[t] = self.risk(w)
            er_tab[t] = self.empirical_risk(w, data)

        gen_tab = risk_tab - er_tab

        # logger.debug(np.linalg.norm(w)/np.sqrt(self.d))

        return gen_tab, risk_tab
    
    def run_one_simulation(self, 
                           data: np.ndarray = None,
                           data_proxy: np.ndarray = None,
                           w0: np.ndarray = None, 
                           alpha_min: float = 1.001, 
                           alpha_max: float = 2.,
                            n_alpha: int = 100):
                
        alpha_tab = np.linspace(alpha_min, alpha_max, n_alpha)

        gen_T_tab = np.zeros(n_alpha)
        gen_sup_tab = np.zeros(n_alpha)

        # Generate dataset one time
        if data is None:
            logger.warning(
                f"No data input for one simulation, generating it.")
            data = self.generate_data()
            data_proxy = self.generate_data_proxy(data)
        assert data.shape == (self.n, self.d)
        assert data_proxy.shape == (self.d, self.d)

        # Starting_point
        if w0 is None:
            logger.warning("No initial point given")
            w0 = self.initialization()
        else:
            w = w0.copy()
        assert w0.ndim == 1
        assert w0.size == self.d

        for k in range(n_alpha):

            gen_tab, _ = self.run_one_alpha_simulation(alpha_tab[k], \
                                                       w0=w0,
                                                        data=data,\
                                                        data_proxy=data_proxy)
            gen_T_tab[k] = gen_tab[self.horizon]
            gen_sup_tab[k] = np.max(gen_tab)

        return alpha_tab, gen_T_tab, gen_sup_tab
    
    def run_simulations_one_dataset(self,
                        n_exp: int = 10,
                        w0: np.ndarray = None,
                         alpha_min: float = 1.001,
                           alpha_max: float = 2.,
                            n_alpha: int = 100):
        
        logger.info(f"Running {n_exp} simulations")


        logger.warning(f"Generating data and data_proxy")
        data = self.generate_data()
        data_proxy = self.generate_data_proxy(data)

        assert data.shape == (self.n, self.d)
        assert data_proxy.shape == (self.d, self.d)

        # Starting point, shared among the simulations
        if w0 is None:
            logger.warning("No initial point given")
            w = self.initialization()
        else:
            w = w0.copy()
        assert w.ndim == 1
        assert w.size == self.d

        gen_T_tab_list = []
        gen_sup_tab_list = []

        for _ in tqdm(range(n_exp)):

            alpha_tab, gen_T_tab, gen_sup_tab =\
                  self.run_one_simulation(data,\
                                          data_proxy,
                                          w0,
                                          alpha_min,
                                          alpha_max,
                                          n_alpha)
            
            gen_T_tab_list.append(gen_T_tab[np.newaxis, ...])
            gen_sup_tab_list.append(gen_sup_tab[np.newaxis, ...])

        gen_T = np.concatenate(gen_T_tab_list, 0)
        gen_sup = np.concatenate(gen_sup_tab_list, 0)

        assert gen_T.shape == (n_exp, n_alpha), gen_T.shape
        assert gen_sup.shape == (n_exp, n_alpha), gen_sup.shape

        return alpha_tab, gen_T, gen_sup
    
    def run_simulations(self,
                        n_dataset: int = 10,
                        n_exp: int = 10,
                         alpha_min: float = 1.001,
                           alpha_max: float = 2.,
                            n_alpha: int = 100):
        
        logger.info(f"Running simulations on {n_dataset} datasets")

        info_dict = {"n": self.n,
                        "d": self.d,
                        "eta": self.eta,
                        "horizon": self.horizon,
                        "total time": self.horizon * self.eta,
                        "Nb of datasets": n_dataset,
                        "Nb of exp.": n_exp}
        logger.info(
            f"Experiment parameters: {json.dumps(info_dict, indent=2)}")

        # Starting point, shared among the simulations
        w0 = self.initialization()

        gen_T_tab_list = []
        gen_sup_tab_list = []

        for _ in tqdm(range(n_dataset)):

            alpha_tab, gen_T_tab, gen_sup_tab =\
                self.run_simulations_one_dataset(n_exp,
                                                    w0,
                                                    alpha_min,
                                                    alpha_max,
                                                    n_alpha)

            gen_T_tab_list.append(gen_T_tab)
            gen_sup_tab_list.append(gen_sup_tab)

        gen_T = np.concatenate(gen_T_tab_list, 0)
        gen_sup = np.concatenate(gen_sup_tab_list, 0)

        assert gen_T.shape == (n_exp * n_dataset, n_alpha), gen_T.shape
        assert gen_sup.shape == (n_exp * n_dataset, n_alpha), gen_sup.shape

        return alpha_tab, gen_T, gen_sup
        


        

    def plot_results_one_simulation(self,
                                alpha_tab: np.ndarray, 
                               gen_T_tab: np.ndarray,
                               gen_sup_tab: np.ndarray,
                               output_name: str = None):
        
        assert alpha_tab.ndim == 1
        assert gen_sup_tab.ndim == 1
        assert gen_T_tab.ndim == 1
        
        assert alpha_tab.size == gen_sup_tab.size
        assert alpha_tab.size == gen_T_tab.size

        if output_name is None:
            output_name = f"{self.experiment_type()}_n_{self.n}_d_{self.d}"
        
        output_path = (self.save_dir / output_name).with_suffix(".png")
        logger.info(f"Saving results in {str(output_path)}")

        plt.figure()

        plt.plot(alpha_tab, gen_T_tab, label='Final generalization')
        plt.plot(alpha_tab, gen_sup_tab, label='Supremum generalization')

        plt.yscale("log")
        
        plt.legend()
        plt.savefig(str(output_path))






        

        



    




    

