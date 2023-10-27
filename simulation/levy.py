import numpy as np
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
    
    assert (1. < alpha and alpha <= 2.), "alpha is expected to be in (1,2]"

    # Handling the case alpha=2, then it is just a scaled Brownian motion
    # Warning: don't forget the sqrt(2) scaling
    if alpha == 2.:
        L = np.random.normal(0., np.sqrt(2), size=(n_iter, d))
    
    # Alpha in (1,2)
    else:
        
        phi = levy_stable.rvs(alpha/2, 1, loc=loc,\
                               scale=2. * np.cos(np.pi*alpha/4)**(2/alpha),\
                                  size=(n_iter, 1)) 
        
        assert (phi >= 0.).all(), "skewed levy process should always be positive"
        L = np.sqrt(phi) * np.random.randn(n_iter, d)

    L = np.power(eta, 1. / alpha) * L

    assert L.shape == (n_iter, d),\
        f"Shape of the returned levy increments shoul be (n_iter, d), got {L.shape}"
    return L

    
