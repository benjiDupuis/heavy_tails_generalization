import numpy as np
import torch

@torch.no_grad()
def sample_standard_gaussian_mixture(dimension: int,
                        n_per_class: int,
                        n_classes: int = 2,
                        means_std: float = 25,
                        blobs_std: float = 100.,
                        random_centers: bool = True,
                        means_deterministic: np.ndarray = None) -> (torch.Tensor, torch.Tensor):
    
    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # Generate means of the blobs
    if random_centers:
        if n_classes > 4:
            means = np.array([
                means_std * np.random.normal(0., 1., size = dimension)\
                    for _ in range(n_classes)
            ])
        else:
            mean_temp = np.random.normal(0., 1., size = dimension)
            mean_temp = mean_temp / np.linalg.norm(mean_temp)
            means = np.array([
                0.5 * means_std * mean_temp,
                -0.5 * means_std * mean_temp
            ])
        means = torch.tensor(means)       
    else:
        assert means_deterministic is not None
        assert means_deterministic.shape == (n_classes, dimension), means_deterministic.shape

        means = means_deterministic.clone()
        # assert n_classes == 2, \
        #     "if centers are determinisitc, number of classes must be 2"
        # means = np.array([
        #     np.sqrt(dimension) * means_std * np.ones(dimension),
        #     (-1.) * np.sqrt(dimension) * means_std * np.ones(dimension)
        # ])

    # Generate each blob
    # WARNING: it only works with scale standard gaussians 
    blobs = []
    labels = []
    for i in range(n_classes):
        blobs.append(means[i].unsqueeze(0) +\
                      blobs_std * torch.randn(size=(n_per_class, dimension)))    
        labels.append(i * torch.ones(n_per_class, dtype=torch.int64))

    # concatenate and random shuffle
    x = torch.concatenate(blobs)
    assert x.shape == (n_per_class * n_classes, dimension), x.shape
    y = torch.concatenate(labels)
    assert y.ndim == 1
    assert y.shape[0] == n_per_class * n_classes, y.shape[0]

    indices = list(np.arange(n_per_class * n_classes))
    np.random.shuffle(indices)

    return x.to(device).float()[indices, ...], y.to(device)[indices], means.to(device)
