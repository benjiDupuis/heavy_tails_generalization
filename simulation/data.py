import numpy as np

def gaussian_iid_risk(w: np.ndarray):

    assert w.ndim == 1, "w is expected to be of size 1"
    return 0.5 * (w * w).sum()


def generate_data_gaussian_iid(n: int, d: int) -> np.ndarray:

    return np.random.normal(0., 1., size=(n, d))
