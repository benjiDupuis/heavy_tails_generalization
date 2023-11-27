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
    

    