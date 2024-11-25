import numpy as np
import pytest

# import required modules
from src.iterative_matrix_augmenters import update_measurement, update_variable, givens_rotation, _compute_sin_and_cos_phi


# Set up test data
m = 10
n = 4
meas_dim = 2
tol = 1e-05

np.random.seed(20)
A = np.random.random(size=(m,n))
Q, R = np.linalg.qr(A) 

b = np.random.random(size=(m,1))
d = Q.T @ b
w_T = np.random.random(size=(meas_dim,n))
gamma = np.random.random(size=(meas_dim,1))


def _compute_full_factorization_with_measurement_update():
    A_p = np.vstack((A, w_T))
    b_p = np.vstack((b, gamma))

    Qp, Rp = np.linalg.qr(A_p)
    dp = Qp.T @ b_p
    return Rp, dp

def test_update_measurement():
    # Compute the true solution
    R_true, d_true = _compute_full_factorization_with_measurement_update()

    R_computed, d_computed = update_measurement(w_T, gamma, R, d)

    print(np.round(R_true, 4))
    print(np.round(R_computed, 4))
    print(np.round(d_true, 4))
    print(np.round(d_computed, 4))

    assert np.isclose(np.abs(R_true), np.abs(R_computed), atol=tol).all()
    assert np.isclose(np.abs(d_true), np.abs(d_computed), atol=tol).all()

def test_update_variable():
    dim = 1
    w_T_big = np.hstack((w_T, np.random.random(size=(w_T.shape[0], dim))))

    R_computed, d_computed = update_variable(dim, w_T_big, gamma, R, d)

    # Check that the size of R is correct
    assert R_computed.shape == tuple(np.array(R.shape) + dim)

    # Check that the solution is correct
    # TODO: 

def test_givens_rotations():
    i = 1
    k = 0

    R_prime, d_prime = givens_rotation(i, k, R, d)
    assert np.isclose(R_prime[i,k], 0.0)

def test_find_cos_and_sin():
    alpha = 2
    beta = 5
    phi = np.atan2(beta, alpha)

    cphi, sphi = _compute_sin_and_cos_phi(alpha, beta)

    assert np.isclose(abs(sphi), abs(np.sin(phi)))
    assert np.isclose(abs(cphi), abs(np.cos(phi)))


if __name__=='__main__':
    test_update_measurement()
