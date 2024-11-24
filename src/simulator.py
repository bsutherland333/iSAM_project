import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Callable

class Simulator:
    """
    Simulator class for the robot. Simulates the robot moving through a 2D world with unknown
    landmark positions but known landmark IDs.
    """
    def __init__(self, inverse_odometry_model: Callable, initial_state: NDArray, dt=1.0,
                 landmark_locations=None, max_range=5.0, max_bearing=np.pi/4):
        """
        Initializes the simulator.

        Parameters:
        inverse_odometry_model: The inverse odometry model of the robot. Should take in the current
            state and the next state and return the odometry.
        dt: The size of the time step for the simulator.
        initial_state: The initial state of the robot. A 3x1 float numpy array [[x, y, theta]].T.
        landmark_locations: The locations of the landmarks in the world.
        max_range: The maximum range of the sensor.
        max_bearing: The maximum bearing of the sensor, measured from the center of the robot's FOV.
        """
        self.inverse_odometry_model = inverse_odometry_model
        self.x = initial_state
        self.dt = dt
        if landmark_locations is None:
            self.landmark_locations = np.array([[0, 0], [1, 1], [2, 2]]).T
        self.max_range = max_range
        self.max_bearing = max_bearing

    def get_next_timestep(self) -> Tuple[NDArray, NDArray]:
        """
        Steps the simulator forward by one time step.

        Returns:
        Numpy array of the odometry reading. Is a 3x1 float array, [[rot1, dist, rot2]].T
        Numpy array of the current measurements. Is a 3xN float array where each measurement is a
            column in the array, [[range, bearing, ID], ...].T.
        """

        return np.array([0, 0, 0]).reshape(-1, 1), np.array([[0, 0], [0, 0], [0, 0]]).T



