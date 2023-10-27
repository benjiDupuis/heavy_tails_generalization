import numpy as np

from simulation.experiments import Experiment


class GaussianExperiment(Experiment):

    def experiment_type(self) -> str:
        return "gaussian_iid_experiment"
    
    def initialization(self) -> np.ndarray:
        return np.zeros(self.d)
    
    def loss(self, w: np.ndarray, data: np.ndarray) -> float:

        assert w.size == data.size, "w and data should have the same size"
        assert w.size == self.d, f"parameters shoud be of size {self.d}"
        assert w.ndim == 1, "w and data are expected to be of dim 1"

        return 0.5 * (w * data).sum()**2
    
    def risk(self, w: np.ndarray) -> float:

        assert w.ndim == 1, "w is expected to be of size 1"
        assert w.size == self.d, f"parameters shoud be of size {self.d}"

        return 0.5 * (w * w).sum()

    def generate_data(self) -> np.ndarray:
        return np.random.normal(0., 1., size=(self.n, self.d))

    def generate_data_proxy(self, data: np.ndarray) -> np.ndarray:

        data_proxy = (data.T @ data) / self.n
        assert data_proxy.shape == (self.d, self.d)

        return data_proxy
        
    def empirical_risk(self, w: np.ndarray, data: np.ndarray) -> float:
        
        assert w.ndim == 1, "w and z are expected to be of dim 1"
        assert data.ndim == 2, "w and z are expected to be of dim 1"

        assert data.shape[0] == self.n
        assert data.shape[1] == self.d
        assert w.size == self.d

        outputs = np.einsum('j,ij->i', w, data)

        assert outputs.ndim == 1
        assert outputs.size == self.n, "number of inputs does not correspond to number of outputs"

        return 0.5 * (outputs * outputs).sum() / self.n
    
    def gradient_empirical_risk(self, w: np.ndarray, data_proxy: np.ndarray) -> np.ndarray:
        
        assert w.ndim == 1, "w and z are expected to be of dim 1"
        assert data_proxy.ndim == 2, "w and z are expected to be of dim 1"
        assert w.size == self.d
        assert data_proxy.shape == (self.d, self.d)
        
        gradient = data_proxy @ w
        assert gradient.ndim == 1
        assert gradient.size == w.size

        # return gradient
        return np.zeros(self.d)
    
