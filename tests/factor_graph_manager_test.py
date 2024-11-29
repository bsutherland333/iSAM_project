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
    print(F(x, u).reshape(3,3))
    assert F(x, u).reshape(3,3).shape == (3, 3)