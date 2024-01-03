import numpy as np
import torch
from loguru import logger
from scipy.special import gamma


class LogisticLoss():

    def __call__(y: torch.Tensor, \
                  target: torch.Tensor) -> torch.Tensor:
        return logistic_loss(y, target)

def logistic_loss(y: torch.Tensor, \
                  target: torch.Tensor) -> torch.Tensor:
    
    """
    the target is supposed to come from the gaussian mixture, ie be either 0 or 1
    """
    assert y.shape == target.shape, (y.shape, target.shape)
    assert y.ndim == 1, y.ndim

    normalized_target = 2 * target - 1
    temp = 1. + torch.exp((-1.) * normalized_target * y)

    # We average over the batch
    return torch.log(temp).mean()


@torch.no_grad()
def accuracy(y: torch.Tensor, \
              target: torch.Tensor) -> torch.Tensor:
    
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
                          threshold: float = 1.e-6) -> float:
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
    


def robust_mean(tab: np.ndarray,
                quantile_up: float = 0.15,
                quantile_low: float = 0.) -> float:
     
    assert quantile_up >= 0. and quantile_up < 0.5, quantile_up
    assert quantile_low >= 0. and quantile_low < 0.5, quantile_low

    assert tab.ndim == 1, tab.shape

    low_quantile = np.quantile(tab, quantile_low)
    high_quantile = np.quantile(tab, 1. - quantile_up)

    indices = (tab >= low_quantile) * (tab <= high_quantile)

    return tab[indices].mean()



def poly_alpha(alpha: float) -> float: 

    if alpha == 2.:
        # asymptotic development of gamma(1-s)
        # using Euler reflection formula
        # TODO: recheck this
        num_alpha = 2.
    else:
        num_alpha = (2. - alpha) * gamma(1. - alpha / 2.)

    den_alpha = alpha * np.power(2., alpha / 2.)

    return num_alpha / den_alpha



def all_linear_regression(
                        gen_grid: np.ndarray,
                        sigma_tab: np.ndarray,
                        alpha_tab: np.ndarray,
                        sigma_low: float = 0.) -> (np.ndarray, np.ndarray):
        """
        Returns the regression of the gen with respect to log(1/sigma), for each alpha
        and the regression of the gen with respect to alpha, for each sigma
        """
        n_alpha = len(alpha_tab)
        n_sigma = len(sigma_tab)

        # Regression gen/log(1/sigma)
        alpha_reg = np.zeros(n_alpha)
        for a in range(n_alpha):
            indices = (gen_grid[:, a] > 0.) * (sigma_tab > sigma_low)

            corrected_gen = np.log(gen_grid[indices, a])
            reg = linear_regression(np.log(1./sigma_tab[indices]), corrected_gen)
            alpha_reg[a] = 2. * reg if reg is not None else None

        # Regression gen/alpha
        correlation_reg = np.zeros(n_sigma)
        for s in range(n_sigma):
            correlation_reg[s] = linear_regression(alpha_tab, gen_grid[s, :])

        return alpha_reg, correlation_reg