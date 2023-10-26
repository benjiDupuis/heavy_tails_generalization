import fire
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from tqdm import tqdm

from quadratic import quadratic_er, quadratic_er_gradient


def gaussian_iid_risk(w: np.ndarray):

    assert w.ndim == 1, "w is expected to be of size 1"
    return 0.5 * (w * w).sum()


def generate_data_gaussian_iid(n: int, d: int) -> np.ndarray:

    return np.random.normal(0., 1., size = (n,d))


def simulate_stable_components(
        alpha: float,
        n: int,
        d: int,
        data: np.ndarray,
        delta: float = 0.01,
        horizon: int = 100,
        sigma: float = 1.,
        starting_point_mean: float = 0.,
        starting_point_std: float = 1.,
        seed: int = 42
    ):
    """
    For now we just handle the gaussian iid case and starting at the origin

    delta: time step
    sigma: noise scale
    """

    np.random.seed(seed)

    assert data.shape == (n, d)

    risk_tab = np.zeros(horizon + 1)
    er_tab = np.zeros(horizon + 1)

    # Starting point
    w = np.random.normal(starting_point_mean, starting_point_std, size=d)

    # Computing the initial values of risk and empirical risk
    risk_tab[0] = gaussian_iid_risk(w)
    er_tab[0] = quadratic_er(w, data)

    A = (data.T @ data) / n
    assert A.shape == (d, d)

    for t in range(1, horizon + 1):

        w = w - delta * quadratic_er_gradient(w, A) + levy_stable.rvs(alpha,
                                                                      0.,
                                                                      0.,
                                                                      sigma * np.power(delta, 1./alpha))

        risk_tab[t] = gaussian_iid_risk(w)
        er_tab[t] = quadratic_er(w, data)

    gen_tab = risk_tab - er_tab

    return gen_tab

def simulate_alpha_iid_gaussian(n: int=100,
                    d: int = 100,
                    n_alpha: int = 100,
                    alpha_min: float = 1.001,
                    alpha_max: float = 2,
                    delta: float = 0.01,
                    horizon: int = 1000,
                    sigma: float = 1.,
                    starting_point_mean: float = 0.,
                    starting_point_std: float = 1.,
                    seed: int = 42):
    
    # Generate data
    S = np.random.normal(0., 1., size = (n,d))

    alphas = np.linspace(alpha_min, alpha_max, n_alpha)
    gen_T_tab = np.zeros(n_alpha)
    gen_sup_tab = np.zeros(n_alpha)

    for k in tqdm(range(n_alpha)):

        gen_tab = simulate_stable_components(alphas[k], n, d, S, delta,
                    horizon,
                    sigma,
                    starting_point_mean,
                    starting_point_std,
                    seed)
        
        gen_T_tab[k] = gen_tab[horizon]
        gen_sup_tab[k] = np.max(gen_tab)

    plt.figure()
    plt.plot(alphas, gen_T_tab, label='Final generalization')
    plt.plot(alphas, gen_sup_tab, label='Supremum generalization')
    plt.legend()
    plt.savefig("figures/test.png")


if __name__ == "__main__":
    fire.Fire(simulate_alpha_iid_gaussian)



    



    
    


