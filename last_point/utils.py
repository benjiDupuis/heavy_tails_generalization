import numpy as np
import torch
from loguru import logger
from scipy.special import gamma



@torch.no_grad()
def accuracy(y: torch.Tensor, \
              target: torch.Tensor) -> torch.Tensor:
    '''
    Computation of the accuracy error from the output of the network
    '''
    
    assert y.shape[0] == target.shape[0], (y.shape[0], target.shape[0])
    assert target.ndim == 1, target.ndim
    assert torch.max(target) < y.shape[1]
    assert y.ndim == 2
    
    estimated_classes = y.argmax(dim=1)
    assert estimated_classes.shape == target.shape,\
                (estimated_classes, target.shape)
    
    return (target == estimated_classes).to("cpu").float().mean()


def linear_regression(x_tab: np.ndarray, 
                          y_tab: np.ndarray, 
                          threshold: float = 1.e-9) -> float:
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

def regression_selection(x_tab: np.ndarray, 
                          y_tab: np.ndarray, 
                          threshold: float = 1.e-9) -> float:

    degrees = np.linspace(0.01, 1., 100)
    r_values = np.zeros(100)

    for k in range(100):

        deg = degrees[k]
        x_tab_pow = np.power(x_tab, deg)
        n = len(x_tab)

        num = (x_tab_pow * y_tab).sum() - x_tab_pow.sum() * y_tab.sum() / n
        den = (x_tab_pow * x_tab_pow).sum() - x_tab_pow.sum()**2 / n

        if den < threshold:
            logger.warning("Inifnite or undefined slope")
            return None

        slope = num / den

        intercept = (y_tab - slope * x_tab_pow).mean()

        r_values[k] = ((y_tab - (slope * x_tab_pow + intercept))**2).sum()
        # logger.info(r_values[k])

    return degrees[np.argmin(r_values)]

        




def robust_mean(tab: np.ndarray,
                quantile_up: float = 0.0,
                quantile_low: float = 0.0) -> float:
    ''''
    Function used to compute a robust mean of the accuracy error over the last iterations
    '''
     
    assert quantile_up >= 0. and quantile_up < 0.5, quantile_up
    assert quantile_low >= 0. and quantile_low < 0.5, quantile_low

    assert tab.ndim == 1, tab.shape

    if len(tab) < 4:
        return tab.mean()

    low_quantile = np.quantile(tab, quantile_low)
    high_quantile = np.quantile(tab, 1. - quantile_up)

    indices = (tab >= low_quantile) * (tab <= high_quantile)

    return tab[indices].mean()


def vector_robust_mean(tab: np.ndarray, quantile: float = 0.1):
    """
    tab should be 1d, of len n_seed
    """ 
    if quantile > 0.:
        est_tab = tab[
            (tab > np.quantile(tab, quantile)) *\
            (tab < np.quantile(tab, 1. - quantile))
        ]
    else:
        est_tab = tab.copy()
    n = len(est_tab)
    m = est_tab.mean()
    c = est_tab - m
    s = np.sqrt(np.power(c, 2).sum() / (n-1))

    return m, s





def matrix_robust_mean(tab: np.ndarray, quantile: float = 0.):
    """
    Used to compute means and standard deviations on some experiments
    """
    n_hyp = tab.shape[0]
    means = np.zeros(n_hyp)
    devs = np.zeros(n_hyp)

    for k in range(n_hyp):
        temp_tab = tab[k,:] 
        est_tab = temp_tab[
            (temp_tab >= np.quantile(temp_tab, quantile)) *\
            (temp_tab <= np.quantile(temp_tab, 1. - quantile))
        ]
        n = len(est_tab)
        m = est_tab.mean()
        c = est_tab - m
        s = np.sqrt(np.power(c, 2).sum() / (n-1))

        means[k] = m
        devs[k] = s

    return means, devs





def poly_alpha(alpha): 
    '''
    Compute the factor P_alpha defined in the paper, in the high dimensional limit
    '''

    if type(alpha) == float and alpha == 2.:
        # asymptotic development of gamma(1-s)
        # using Euler reflection formula
        # TODO: recheck this
        num_alpha = 2.
    else:
        num_alpha = (2. - alpha) * gamma(1. - alpha / 2.)

    den_alpha = alpha * np.power(2., alpha / 2.)

    return num_alpha / den_alpha

