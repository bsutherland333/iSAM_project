import numpy as np
import jax.numpy as jnp
import pytest

from src import factor_graph_manager as fgm
from src.jax_jacobians import make_F_H_J_G
from src.models import motion_model, sensor_model, inverse_motion_model


def test_F_jacobian():
    F, _, _, G = make_F_H_J_G(motion_model, sensor_model)
    x = jnp.array([np.cos(0.), np.sin(0.), 0.]).reshape(3, 1)
    u = jnp.array([0., 0., 0.]).reshape(3, 1)
    expected = np.eye(3)
    assert F(x, u).reshape(3,3).shape == (3, 3)
    assert np.allclose(F(x, u).reshape(3,3), expected)

def test_H_jacobian():
    F, H, J, G = make_F_H_J_G(motion_model, sensor_model)
    x = jnp.array([1, 1, 0.]).reshape(3, 1)
    l = jnp.array([2.,2.]).reshape(2, 1)
    H_eval = H(x, l).reshape(2,3)
    H_expected = np.array([[-np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                           [1/2, -1/2, -1]])
    assert np.allclose(H_eval, H_expected)

def test_J_jacobian():
    F, H, J, G = make_F_H_J_G(motion_model, sensor_model)
    x = jnp.array([1, 1, 0.]).reshape(3, 1)
    l = jnp.array([2.,2.]).reshape(2, 1)
    J_eval = J(x,l).reshape(2,2)
    J_expected = np.array([[np.sqrt(2)/2, np.sqrt(2)/2],
                           [-1/2, 1/2]])
    assert np.allclose(J_eval, J_expected)

def test_G_jacobian():
    F, H, J, G = make_F_H_J_G(motion_model, sensor_model)
    x = jnp.array([1, 1, 0.]).reshape(3, 1)
    G_eval = G(x)
    G_expected = -np.eye(3)
    assert np.allclose(G_eval, G_expected)