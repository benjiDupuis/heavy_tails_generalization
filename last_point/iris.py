import numpy as np
import torch
from sklearn import datasets



@torch.no_grad()
def sample_iris_dataset(train_proportion: float = 0.8):

    assert train_proportion > 0.
    assert train_proportion < 1.

    raw_data = datasets.load_iris()
    n = raw_data["data"].shape[0]
    d = raw_data["data"].shape[1]

    # shuffling
    idx = np.arange(n)
    np.random.shuffle(idx)
    
    # Apply it
    x = raw_data["data"][idx]
    y = raw_data["target"][idx]

    # construct train and val
    n_train = int(train_proportion * n)
    x_train = x[:n_train, :]
    y_train = y[:n_train]
    x_val = x[n_train:, :]
    y_val = y[n_train:]

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # return as tensors
    return (
        torch.tensor(x_train).to(device).float(),
        torch.tensor(y_train).to(device),
        torch.tensor(x_val).to(device).float(),
        torch.tensor(y_val).to(device),
    ), (n_train, d)
