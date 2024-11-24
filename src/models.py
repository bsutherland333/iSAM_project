import numpy as np
from numpy.typing import NDArray


def odometry_model(x: NDArray, u: NDArray) -> NDArray:
    """
    Odometry model for our system. Uses basic "rotate, move, rotate" model from chapter 5.4 of
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

    return np.array([0, 0, 0], dtype=float).reshape(3, 1)


def inverse_odometry_model(x: NDArray, x_next: NDArray) -> NDArray:
    """
    Inverse odometry model of our system. Like the odometry model, but instead of getting the next
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

    return np.array([0, 0, 0], dtype=float).reshape(3, 1)


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

    return np.array([0, 0], dtype=float).reshape(2, 1)

