import numpy as np
from numpy.typing import NDArray

from scipy.linalg import solve_triangular


def np_solver(R: NDArray, d: NDArray) -> NDArray:
    """
    Solve the linear system R @ x = d for x. Basic wrapper around numpy's solve function.

    Parameters:
    R: The R matrix from the QR decomposition. (NxN)
    d: The d = Q.T @ b vector from the linear system. (Nx1)

    Returns:
    The x vector that solves the linear system. (Nx1)
    """
    assert R.ndim == 2
    assert R.shape[0] == R.shape[1]
    assert R.dtype == np.float64
    assert d.shape == (R.shape[0], 1)
    assert d.dtype == np.float64

    return np.linalg.solve(R, d)


# Place other methods here with same input/output
def scipy_solver(R: NDArray, d: NDArray) -> NDArray:
    """
    Solve the linear system R @ x = d for x. Basic wrapper around scipy's solve_triangular function.

    Parameters:
    R: The R matrix from the QR decomposition. (NxN)
    d: The d = Q.T @ b vector from the linear system. (Nx1)

    Returns:
    The x vector that solves the linear system. (Nx1)
    """
    assert R.ndim == 2
    assert R.shape[0] == R.shape[1]
    assert R.dtype == np.float64
    assert d.shape == (R.shape[0], 1)
    assert d.dtype == np.float64

    return solve_triangular(R, d)
