import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jacrev
import jax
from typing import Callable, Tuple
from numpy.typing import NDArray

# this is to make all the arrays float64
jax.config.update("jax_enable_x64", True)

def make_F_H_J_G(motion_model: Callable, measurement_model: Callable) -> Tuple[Callable, Callable, Callable, Callable]:
    F = jacfwd(motion_model, argnums=0)
    G = lambda x : -np.eye(len(x))
    H = jacfwd(measurement_model, argnums=0)
    J = jacfwd(measurement_model, argnums=1)

    return F, H, J, G

