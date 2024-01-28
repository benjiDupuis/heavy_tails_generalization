from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy.stats import levy_stable


def simulate_levy(d, alp, eta):
    T = np.ceil(1.0/eta)
    phi = levy_stable.rvs(alp/2, 1, loc=0, scale=2 *
                          np.cos(np.pi*alp/4)**(2/alp), size=(1, int(T)))
    L = np.sqrt(phi) * np.random.randn(d, int(T))
    L = L * (eta**(1/alp))
    W = np.cumsum(L, axis=1)
    return W


def generate_levy_for_simulation(d: int,
                                 n_iter: int,
                                 alpha: float,
                                 eta: float,
                                 loc: float = 0.) -> np.ndarray:
    """
    d: ambiant dimension
    n_iter: number of iterations
    alpha: tail index
    eta: learning rate (to correctly scale the time step)
    loc: "mean", set to 0

    IMPORTANT: the time scale eta**(1/alpha) is already taken into 
    account in this function
    """
    
    assert (1. < alpha and alpha <= 2.), "alpha is expected to be in (1,2]"

    # Handling the case alpha=2, then it is just a scaled Brownian motion
    # Warning: don't forget the sqrt(2) scaling
    if alpha == 2.:
        L = np.random.normal(0., np.sqrt(2), size=(n_iter, d))
    
    # Alpha in (1,2)
    else:
        
        phi = levy_stable.rvs(alpha/2., 1, loc=loc,\
                               scale=2. * np.cos(np.pi*alpha/4.)**(2./alpha),\
                                  size=(n_iter, 1)) 
        
        assert (phi >= 0.).all(), "skewed levy process should always be positive"
        # L = np.sqrt(phi) * np.random.randn(n_iter, d)
        L = np.sqrt(phi) * np.random.randn(d, n_iter).T

    L = np.power(eta, 1. / alpha) * L

    assert L.shape == (n_iter, d),\
        f"Shape of the returned levy increments should be (n_iter, d), got {L.shape}"
    return L

    

def plot_levy(n_iter: int = 1000,
              alpha: float = 2.,
              eta: float = 0.001,
              output_dir: str = "levy_plots"):
    
    assert n_iter >= 2, n_iter
    
    # Simulate it
    noise = generate_levy_for_simulation(2, n_iter, alpha, eta)
    assert noise.shape == (n_iter, 2), noise.shape

    point = np.zeros(2)

    plt.figure()

    for k in range(0, n_iter):
        next_point = point + noise[k, :]
        plt.plot([point[0], next_point[0]],\
                 [point[1], next_point[1]],
                   color="k")
        point += noise[k, :]      
        
    plt.title(f"Levy process simulation for alpha = {alpha}")

    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (output_dir / f"alpha_{alpha}".replace(".", "_")).with_suffix(".png")
    logger.info(f"Saving figure in {str(output_path)}")
    plt.savefig(str(output_path))


def main(alpha: float = 1.8,
         eta: float = 0.001,
         output_dir: str = "levy_plots",
         seed: int = 1):
    
    np.random.seed(seed)
    plot_levy(1000, alpha, eta, output_dir)

if __name__ == "__main__":
    fire.Fire(main)



    

    

