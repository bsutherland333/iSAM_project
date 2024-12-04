import numpy as np
from numpy.typing import NDArray
from typing import Callable

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


# Nonlinear solver
def iterative_linear_least_squares(x: NDArray, get_A_b: Callable, reorder: Callable) -> tuple[NDArray, NDArray, NDArray]:
    """
    Solves the equation Ax = b iteratively to find the solution to the nonlinear equations from which
        A and b are derived.

    Parameters:
    x: The current value of the solution. (Nx1)
    get_A_b: Callable that computes the A and b matrices given x
    reorder: Callable that reorders the columns of A to preserve sparsity

    Returns:
    x: The solution to Ax=b after the solution converges. (Nx1)
    R: The final factorized R matrix. (NxN)
    d: The final d vector. (Nx1)
    """
    assert x.ndim == 2
    assert x.dtype == np.float64
    assert x.shape[1] == 1

    threshold = 1.0
    solved_once = False
    prev_x = np.zeros(x.shape)

    while np.linalg.norm(x - prev_x) > threshold or not solved_once:
        solved_once = True
        prev_x = x.copy()

        A, b = get_A_b(x)
        A_prime, P = reorder(A)

        # Factor A
        Q, R = np.linalg.qr(A_prime)
        d = Q.T @ b

        #x_prime = np.linalg.lstsq(A_prime, b)[0]
        x_prime = solve_triangular(R, d)
        x += P @ x_prime

    return x, R, d


# Explicitly coded solver for the sake of the project.
def solver_homemade(R: NDArray, d: NDArray) -> NDArray:
    """
    Solve the linear system R @ x = d for x.

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

    n = R.shape[0]
    x = np.zeros((n, 1))

    for i in range(n-1, -1, -1):
        x[i] = (d[i] - np.dot(R[i, i+1:], x[i+1:])) / R[i, i]

    return x

