import numpy as np
import torch

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
                quantile: float = 0.15) -> float:
     
    assert quantile > 0. and quantile < 0.5, quantile
    assert tab.ndim == 1, tab.shape

    low_quantile = np.quantile(tab, quantile)
    high_quantile = np.quantile(tab, 1. - quantile)

    indices = (tab >= low_quantile) * (tab <= high_quantile)

    return tab[indices].mean()