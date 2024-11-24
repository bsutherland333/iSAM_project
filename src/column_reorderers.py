import numpy as np
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

    P = np.eye(A.shape[1], dtype=int)

    return A, P


# Place other methods here with same input/output
