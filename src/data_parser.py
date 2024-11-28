import os
import numpy as np


class DataParser:
    """
    Parses the data from a file and returns odometry and sensor data at each timestep.
    """
    def __init__(self, filepath: str):
        """
        Initializes the DataParser.

        Parameters:
        filepath: The absolute path to the data file.
        """
        raise NotImplementedError("DataParser has not been implemented yet.")

    def get_initial_state(self):
        """
        Returns the true initial state of the robot.
        """
        return np.array([0, 0, 0], dtype=float).reshape(3, 1)

    def get_next_timestep(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Gets the odometry and sensor measurements for the next timestep.

        Returns:
        Numpy array of the odometry reading. Is a 3x1 float array, [[rot1, dist, rot2]].T
        Numpy array of the current measurements. Is a 3xN float array where each measurement is a
            column in the array, [[range, bearing, ID], ...].T.
        """

        return np.array([0, 0, 0]).reshape(-1, 1), \
               np.array([[0, 0], [0, 0], [0, 0]]).T, \
               np.array([0, 0, 0]).reshape(-1, 1)
