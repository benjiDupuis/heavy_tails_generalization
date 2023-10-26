import numpy as np


def quadratic_loss(w: np.ndarray, z: np.ndarray):

    assert w.size == z.size, "w and z should have the same size"
    assert w.ndim == 1, "w and z are expected to be of dim 1"

    return 0.5 * (w * z).sum()**2


def quadratic_er(w: np.ndarray, S: np.ndarray):

    assert w.ndim == 1, "w and z are expected to be of dim 1"
    assert S.ndim == 2, "w and z are expected to be of dim 1"
    assert w.size == S.shape[1], "the size of w should match the number of columns of S"

    n = S.shape[0]
    outputs = np.einsum('j,ij->i', w, S)

    assert outputs.ndim == 1
    assert outputs.size == n, "number of inputs does not correspond to number of outputs"

    return 0.5 * (outputs * outputs).sum() / n


def quadratic_er_gradient(w: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    A should be S^T@S/n, to avoid computing it several times
    """

    assert w.ndim == 1, "w and z are expected to be of dim 1"
    assert A.ndim == 2, "w and z are expected to be of dim 1"
    assert w.size == A.shape[0], "w and A should have the same size"
    assert w.size == A.shape[1], "w and A should have the same size"

    gradient = A @ w
    assert gradient.ndim == 1
    assert gradient.size == w.size

    return gradient





    
