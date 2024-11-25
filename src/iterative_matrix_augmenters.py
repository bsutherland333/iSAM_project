import numpy as np

def update_measurement(w_T:np.ndarray, gamma:np.ndarray, R:np.ndarray, d:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Adds a measurement row to the R matrix and zeros it out with Givens rotations 
    as described in Kaess et. al in the iSAM paper.

    Input:  w_T is a (k x n) array representing the new (block) row corresponding to
               the new measurement.
            gamma is a (k x 1) array representing the value of the new measurement
            R is a (n x n) matrix. It is the current Cholesky triangle of A.T @ A
            d is a (n x 1) vector. It is the current array of measurements

    Output: R_prime (n x n) matrix transformed using the Givens rotations procedure
                described by Kaess et. al.
            b_prime (n x 1) vector transformed by the same Givens rotations
    '''
    # Check sizes for compatibility
    assert R.shape[0] == R.shape[1]
    assert w_T.shape[1] == R.shape[1]
    assert w_T.shape[0] == gamma.shape[0]

    if len(gamma.shape) < 2:
        assert gamma.shape[0] == 1
    else:
        assert gamma.shape[1] == d.shape[1]

    R_prime = np.vstack((R, w_T))
    d_prime = np.vstack((d, gamma))

    # Iterate through the row(s) we just added and zero out all the elements
    for i in range(R.shape[0], R_prime.shape[0]):
        for k in range(R.shape[1]):
            if not np.isclose(R_prime[i,k], 0.0):
                R_prime, d_prime = givens_rotation(i, k, R_prime, d_prime)

    # Return only the square portion of the R matrix. The rest should be zero
    return R_prime[:R_prime.shape[1],:], d_prime[:R_prime.shape[1]]


def update_variable(dim:int, w_T:np.ndarray, gamma:np.ndarray, R:np.ndarray, d:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Adds dim empty rows and columns to R and then adds a measurement row to the R matrix
    and zeros it out with Givens rotations using update_measurement.

    Input:  dim is the number of rows and columns to add. Corresponds to the size of the 
                new variable to add.
            w_T is a (k x n) array representing the new (block) row corresponding to
                the new measurement.
            gamma is a (k x 1) array representing the value of the new measurement
            R is a (n x n) matrix. It is the current Cholesky triangle of A.T @ A
            d is a (n x 1) vector. It is the current array of measurements

    Output: R_prime (n+dim x n+dim) matrix transformed using the Givens rotations
                procedure described by Kaess et. al.
            b_prime (n+dim x 1) vector transformed by the same Givens rotations
    '''
    # Check sizes for compatibility
    assert R.shape[0] == R.shape[1]
    assert w_T.shape[1] == (R.shape[1] + dim)
    assert w_T.shape[0] == gamma.shape[0]

    if len(gamma.shape) < 2:
        assert gamma.shape[0] == 1
    else:
        assert gamma.shape[1] == d.shape[1]

    # Append empty vectors of size dim to the R matrix and d vector
    R_prime = np.vstack((
        np.hstack((R, np.zeros((R.shape[0], dim)))),
        np.zeros((dim, R.shape[1] + dim))
    ))

    d_prime = np.vstack((d, np.zeros((dim, d.shape[1]))))


    # Apply the measurement update
    R_prime, d_prime = update_measurement(w_T, gamma, R_prime, d_prime)

    return R_prime, d_prime

def _compute_sin_and_cos_phi(alpha:float, beta:float) -> tuple[float, float]:
    '''
    Computes cos(phi) and sin(phi) without calling the sin and cos functions or 
    explicitly computing phi. Used in the Givens rotation computation

    Input:  alpha is a float value
            beta is also a float value.
    Output: (cos(phi), sin(phi)), a tuple of floats corresponding to cos(phi) and
                sin(phi).

    Example: As defined in Kaess et. al, alpha is the [k,k]th element of the matrix
    and beta is the [i,k]th element.
    '''
    if beta == 0:
        return (1,0)

    if abs(beta) > abs(alpha):
        a = -alpha / (beta*np.sqrt(1+(alpha/beta)**2))
        b = 1 / np.sqrt(1+(alpha/beta)**2)
        return (a, b)

    # Otherwise
    a = 1 / np.sqrt(1+(beta/alpha)**2)
    b = -beta / (alpha*np.sqrt(1+(beta/alpha)**2))
    return (a, b)

def givens_rotation(i:int, k:int, R:np.ndarray, d:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''
    Zeros out the (i, k) element in the matrix R by applying a Givens rotation. 
    Note that i > k

    Input:  i is the index of the row of the element to zero out. i > k
            k is the index of the column of the element to zero out. i > k
            R is an (n x n) matrix. It is the current Cholesky triangle of A.T @ A

    Output: R_prime (n x n) matrix with the (i,k)th element zeroed.
    '''
    assert i < R.shape[0]
    assert k < R.shape[1]
    assert i > k
    assert R.shape[0] == d.shape[0]

    alpha = R[k,k]
    beta = R[i,k]
    cphi, sphi = _compute_sin_and_cos_phi(alpha, beta)

    givens = np.eye(R.shape[0])
    givens[k,k] = cphi
    givens[i,i] = cphi
    givens[i,k] = sphi
    givens[k,i] = -sphi

    return givens @ R, givens @ d
