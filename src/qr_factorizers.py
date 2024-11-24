import numpy as np
from typing import Tuple
from numpy.typing import NDArray


def np_qr(A: np.ndarray) -> Tuple[NDArray, NDArray]:
    """
    A basic wrapper around numpy's QR decomposition function.

    Parameters:
    A: The numpy matrix to decompose.

    Returns:
    The Q and R matrices from the QR decomposition.
    """
    assert A.ndim == 2
    assert A.dtype == np.float64

    Q, R = np.linalg.qr(A)

    return Q, R


# Place other methods here with same input/output
