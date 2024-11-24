import numpy as np
from numpy.typing import NDArray


def np_solver(Q: NDArray, R: NDArray, b: NDArray) -> NDArray:
    """
    Solve the linear system Q @ R @ x = b for x. Basic wrapper around numpy's solve function.

    Parameters:
    Q: The Q matrix from the QR decomposition. (MxN)
    R: The R matrix from the QR decomposition. (NxN)
    b: The b vector from the linear system. (Mx1)

    Returns:
    The x vector that solves the linear system. (Nx1)
    """
    assert Q.ndim == 2
    assert Q.dtype == np.float64
    assert R.shape == (Q.shape[1], Q.shape[1])
    assert R.dtype == np.float64
    assert b.shape == (Q.shape[0], 1)
    assert b.dtype == np.float64

    A = Q @ R

    return np.linalg.solve(A, b)


# Place other methods here with same input/output
