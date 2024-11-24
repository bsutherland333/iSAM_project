import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple


class FactorGraphManager:
    """
    Class to construct the A matrix from a set of given odometry and sensor measurements.
    """
    def __init__(self, odometry_model: Callable, sensor_model: Callable, initial_state: NDArray) -> None:
        """
        Initialize the FactorGraphManager.

        Parameters:
        odometry_model: Callable function of the odometry model we are using. Input should be x and u,
            output should be x.
        sensor_model: Callable function of the sensor model we are using. Input should be x and
            landmark ID, output should be z.
        """
        self.odometry_model = odometry_model
        self.sensor_model = sensor_model
        self.x = initial_state
        self.prev_A_width = initial_state.shape[0]

    def add_measurement(self, z: NDArray) -> None:
        """
        Adds a measurement to the factor graph.

        Parameters:
        z: The measurement to add to the factor graph. A 3x1 float numpy array [[range, bearing,
        ID]].T.
        """
        assert z.shape == (3, 1)
        assert z.dtype == np.float64

    def add_odometry(self, u: NDArray) -> None:
        """
        Adds an odometry measurement to the factor graph.

        Parameters:
        u: The odometry measurement to add to the factor graph.
        """
        assert u.shape == (3, 1)
        assert u.dtype == np.float64

    def get_A_b_matrix(self, x: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Computes the A matrix from the current set of odometry and sensor measurements for a given
        linearization point (best state estimate) x. x is based on the PREVIOUS A matrix, not the
        one to be constructed. b is also constructed and returned.

        Parameters:
        x: The current best estimate of the state. A Nx1 float numpy array, where the state is the
            estimated position and orientation of the robot at all time steps, followed by the
            estimated position of each landmark.

        Returns:
        The A matrix and its associated b vector.
        """
        assert x.shape == (self.prev_A_width, 1)
        assert x.dtype == np.float64

        return np.zeros((9, 6)), np.zeros((9, 1))

