import numpy as np
from scipy.sparse.linalg import splu
from numpy.typing import NDArray
from typing import Tuple


def colamd(A: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Reorders the columns of A using the COLAMD algorithm.

    Parameters:
    A: The matrix to reorder.

    Returns:
    The reordered matrix A' and the permutation matrix P.
    """
    assert A.ndim == 2
    assert A.dtype == np.float64

    p = splu(A.T @ A, permc_spec='COLAMD').perm_c
    P = np.zeros((A.shape[1], A.shape[1]), dtype=int)
    P[np.arange(np.shape(A)[1]), p] = 1

    return A @ P, P


def passthrough(A: NDArray) -> Tuple[NDArray, NDArray]:
    """
    Passthrough function for the reordering algorithm. Does nothing.

    Parameters:
    A: The matrix to reorder.

    Returns:
    The original matrix A and the identity matrix.
    """
    assert A.ndim == 2
    assert A.dtype == np.float64

    return A, np.eye(A.shape[1], dtype=int)
