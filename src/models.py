import numpy as np
import jax.numpy as jnp
from jax import jit
from numpy.typing import NDArray

@jit
def motion_model(x: NDArray, u: NDArray) -> NDArray:
    """
    Motion model for our system. Uses basic "rotate, move, rotate" model from chapter 5.4 of
    Probabilistic Robotics.

    Parameters:
    x: The current state of the system. A 3x1 float numpy array [[x, y, theta]].T.
    u: The control input to the system. A 3x1 float numpy array [[rot1, dist, rot2]].T.

    Returns:
    The next state of the system. A 3x1 float numpy array [[x, y, theta]].T.
    """
    assert x.shape == (3, 1)
    assert x.dtype == np.float64
    assert u.shape == (3, 1)
    assert u.dtype == np.float64

    motion_heading = x[2] + u[0]

    x_next = x[0] + u[1] * jnp.cos(motion_heading)
    y_next = x[1] + u[1] * jnp.sin(motion_heading)
    heading_next = motion_heading + u[2]

    heading_next = _wrap_within_pi(heading_next)

    return jnp.array([x_next, y_next, heading_next], dtype=float).reshape(3, 1)


def inverse_motion_model(x: NDArray, x_next: NDArray) -> NDArray:
    """
    Inverse motion model of our system. Like the motion model, but instead of getting the next
    state given the current and control input, we get the control input given the current and next
    state.

    Parameters:
    x: The current state of the system. A 3x1 float numpy array [[x, y, theta]].T.
    x_next: The next state of the system. A 3x1 float numpy array [[x, y, theta]].T.

    Returns:
    The control input to the system. A 3x1 float numpy array [[rot1, dist, rot2]].T.
    """
    assert x.shape == (3, 1)
    assert x.dtype == np.float64
    assert x_next.shape == (3, 1)
    assert x_next.dtype == np.float64

    del_x = x_next[0] - x[0]
    del_y = x_next[1] - x[1]

    dist = np.sqrt(del_x**2 + del_y**2)
    rot1 = np.arctan2(del_y, del_x) - x[2]
    rot2 = x_next[2] - x[2] - rot1

    rot1 = _wrap_within_pi(rot1)
    rot2 = _wrap_within_pi(rot2)

    return np.array([rot1, dist, rot2], dtype=float).reshape(3, 1)


def sensor_model(x: NDArray, landmark: NDArray) -> NDArray:
    """
    Sensor model for our system. Uses a basic range and bearing model.

    Parameters:
    x: The current state of the system. A 3x1 float numpy array [[x, y, theta]].T.
    landmark: The xy position of the landmark. A 2x1 float numpy array [[x, y]].T.

    Returns:
    The expected measurement to the landmark. A 2x1 float numpy array [[range, bearing]].
    """
    assert x.shape == (3, 1)
    assert x.dtype == np.float64
    assert landmark.shape == (2, 1)
    assert landmark.dtype == np.float64

    del_x = landmark[0] - x[0]
    del_y = landmark[1] - x[1]

    range = jnp.sqrt(del_x**2 + del_y**2)
    bearing = jnp.arctan2(del_y, del_x) - x[2]

    bearing = _wrap_within_pi(bearing)

    return jnp.array([range, bearing], dtype=float).reshape(2, 1)


def _wrap_within_pi(val: float) -> float:
    """
    Wrap a value within the range of -pi to pi.

    Parameters:
    val: The value to wrap.

    Returns:
    The wrapped value.
    """
    return (val + np.pi) % (2 * np.pi) - np.pi
