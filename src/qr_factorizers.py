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


# Explicitly coded qr factorizor for the sake of the project.
def qr_homemade(A: np.ndarray) -> Tuple[NDArray, NDArray]:

    assert A.ndim == 2
    assert A.dtype == np.float64

    m, n = A.shape
    
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        
        # Step 1: Copy the j-th column of A
        v = A[:, j]
        
        # Step 2: Orthogonalize v against the previous columns of Q
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v = v - R[i, j] * Q[:, i]
        
        # Step 3: Normalize the resulting vector to get the j-th column of Q
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R
