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

def test_factor_graph_manager():
    initial_state = jnp.array([0,0,0]).reshape(3,1)
    factor_graph_manager = fgm.FactorGraphManager(initial_state, 3)
    F, H, J, G = make_F_H_J_G(motion_model, sensor_model)

    measurement1 = jnp.array([2,np.pi/4,0]).reshape(3,1)
    factor_graph_manager.add_measurement(measurement1, H, J)

    odometry1 = jnp.array([0,1.,0]).reshape(3,1)
    factor_graph_manager.add_odometry(odometry1, F, G)

    measurement2 = jnp.array([1, np.pi,0]).reshape(3,1)
    factor_graph_manager.add_measurement(measurement2, H, J)
    x = jnp.array([0,0,0,1.,0.,0.,1.5,-1]).reshape(8,1)
    A, b = factor_graph_manager.get_A_b_matrix(x)

    b_expected = np.array([0., 0., 0., 0, 1., 0, 2, np.pi/4, 1, np.pi]).reshape(-1,1)
    x0 = np.array([0.,0.,0.]).reshape(3,1)
    l0 = np.array([1.5, -1]).reshape(2,1)
    F_1_2 = F(x0, odometry1).reshape(3,3)
    H_1_1 = H(x0, l0).reshape(2,3)
    J_1_1 = J(x0, l0).reshape(2,2)

    x1 = np.array([1., 0., 0.]).reshape(-1,1)
    H_1_2 = H(x1, l0).reshape(2,3)
    J_1_2 = J(x1, l0).reshape(2,2)

    A_expected = np.array([[-1, 0, 0, 0, 0, 0, 0, 0],
                           [0, -1, 0, 0, 0, 0, 0, 0],
                           [0, 0, -1, 0, 0, 0, 0, 0],
                           [F_1_2[0,0], F_1_2[0,1], F_1_2[0,2], -1, 0, 0, 0, 0],
                           [F_1_2[1,0], F_1_2[1,1], F_1_2[1,2], 0, -1, 0, 0, 0],
                           [F_1_2[2,0], F_1_2[2,1], F_1_2[2,2], 0, 0, -1, 0, 0],
                           [H_1_1[0,0], H_1_1[0,1], H_1_1[0,2], 0, 0, 0, J_1_1[0,0], J_1_1[0,1]],
                           [H_1_1[1,0], H_1_1[1,1], H_1_1[1,2], 0, 0, 0, J_1_1[1,0], J_1_1[1,1]],
                           [0, 0, 0, H_1_2[0,0], H_1_2[0,1], H_1_2[0,2], J_1_2[0,0], J_1_2[0,1]],
                           [0, 0, 0, H_1_2[1,0], H_1_2[1,1], H_1_2[1,2], J_1_2[1,0], J_1_2[1,1]]])
    
    assert np.allclose(b, b_expected)
    assert np.allclose(A, A_expected)
    

    