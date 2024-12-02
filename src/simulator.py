from enum import Enum
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Callable


class MotionGeometryType(Enum):
    """
    Enum class for specifying what kind of trajectory the robot should follow. We can add more
    trajectories in the future if we need.
    """
    RECTANGLE = "rectangle"


class Simulator:
    """
    Simulator class for the robot. Simulates the robot moving through a 2D world with unknown
    landmark positions but known landmark IDs.
    """
    def __init__(self,
                 inverse_motion_model: Callable,
                 sensor_model: Callable,
                 initial_state:NDArray = np.array([[0.5, 0.5, 0]], float).reshape(-1, 1),
                 motion_type:MotionGeometryType = MotionGeometryType.RECTANGLE,
                 motion_size:Tuple[float, float] = (5.0, 2.0),
                 landmark_locations:NDArray = np.array(
                    [[0, 0], [3, 0], [6, 0], [0, 3], [3, 3], [6, 3]],
                    float
                 ).T,
                 dt:float = 1.0,
                 np_seed:int = 0,
                 ):
        """
        Initializes the simulator.

        Parameters:
        inverse_motion_model: The inverse motion model of the robot. Should take in the current
            state and the next state and return the odometry.
        sensor_model: The sensor model of the robot. Should take in the current state and the
            location of a landmark and return the measurement.
        initial_state: The initial state of the robot. A 3x1 float numpy array [[x, y, theta]].T.
            Units are in meters and radians.
        motion_type: The type of motion geometry to use. Currently only rectangle is supported.
        dt: The size of the time step for the simulator, in seconds.
        landmark_locations: The locations of the landmarks in the world, in meters.
        np_seed: The seed to use for the numpy random number generator. Helps with consistency.
        """
        assert initial_state is None or initial_state.shape == (3, 1)
        assert initial_state is None or initial_state.dtype == np.float64
        assert dt > 0
        assert landmark_locations is None or landmark_locations.shape[0] == 2
        assert landmark_locations is None or landmark_locations.ndim == 2
        assert landmark_locations is None or landmark_locations.dtype == np.float64

        # Passed parameters
        self._inverse_motion_model = inverse_motion_model
        self._sensor_model = sensor_model
        self._initial_state = initial_state.copy()
        self._prev_x = np.zeros((3, 1), float)
        self._motion_type = motion_type
        self._motion_size = motion_size
        self._landmarks = landmark_locations.copy()
        self._dt = dt
        np.random.seed(np_seed)

        self._t = 0
        self._landmark_ids = np.zeros(landmark_locations.shape[1], int) - 1

        # Shift and rotate landmarks to be relative to the initial state (making the initial state
        # the origin)
        self._landmarks -= self._initial_state[0:2]
        theta = self._initial_state.item(2)
        self._R_sim_truth = np.array([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta), np.cos(theta)]])
        self._landmarks = self._R_sim_truth.T @ self._landmarks

        # Sensor parameters
        self._max_range = 5.0
        self._range_std = 0.1
        self._range_bias = 0.0
        self._max_bearing = np.pi/2
        self._bearing_std = 0.1
        self._bearing_bias = 0.0

        # Odometry parameters
        self._odometry_rotation_std = 0.05
        self._odometry_rotation_bias = 0.0
        self._odometry_translation_std = 0.02
        self._odometry_translation_bias = 0.0

    def get_next_timestep(self) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Steps the simulator forward by one time step.

        Returns:
        Numpy array of the odometry reading. Is a 3x1 float array, [[rot1, dist, rot2]].T
        Numpy array of the current measurements. Is a 3xN float array where each measurement is a
            column in the array, [[range, bearing, ID], ...].T.
        Numpy array of the true state of the robot. Is a 3x1 float array, [[x, y, theta]].T.
        """

        # Iterate the sim
        self._t += self._dt

        # Get the next state of the robot
        if self._motion_type == MotionGeometryType.RECTANGLE:
            next_state = self._get_next_rectangle_state()
        else:
            raise Exception("Motion geometry not supported")

        # Get sensor measurements
        measurements = []
        for i in range(self._landmarks.shape[1]):
            landmark = self._landmarks[:, i].reshape(-1, 1)
            measurement = np.array(self._sensor_model(next_state, landmark))

            # Landmark visible, add to measurements
            if measurement[0] <= self._max_range and np.abs(measurement[1]) <= self._max_bearing:
                # Check if id has been assigned
                if self._landmark_ids[i] == -1:
                    self._landmark_ids[i] = np.max(self._landmark_ids) + 1

                measurement[0] += np.random.normal(self._range_bias, self._range_std)
                measurement[1] += np.random.normal(self._bearing_bias, self._bearing_std)

                measurements.append(np.array([measurement.item(0), measurement.item(1), self._landmark_ids[i]], float).reshape(-1, 1))
        measurements = np.hstack(measurements)

        # Get the odometry reading
        odometry = self._inverse_motion_model(self._prev_x, next_state)
        self._prev_x = next_state
        odometry[0] += np.random.normal(self._odometry_rotation_bias, self._odometry_rotation_std)
        odometry[1] += np.random.normal(self._odometry_translation_bias, self._odometry_translation_std)
        odometry[2] += np.random.normal(self._odometry_rotation_bias, self._odometry_rotation_std)

        # Transform from the robot frame to the world frame
        global_state = next_state.copy()
        global_state[:2] = self._R_sim_truth @ global_state[:2]
        global_state += self._initial_state

        return odometry, measurements, global_state

    def _get_next_rectangle_state(self):
        width = self._motion_size[0]
        height = self._motion_size[1]

        loop_dist = 2 * width + 2 * height
        curr_dist = self._t % loop_dist  # Assume velocity is 1 m/s

        if curr_dist < width:
            x = curr_dist
            y = 0
            theta = 0
        elif curr_dist < width + height:
            x = width
            y = curr_dist - width
            theta = np.pi/2
        elif curr_dist < 2 * width + height:
            x = width - (curr_dist - width - height)
            y = height
            theta = -np.pi
        else:
            x = 0
            y = height - (curr_dist - 2 * width - height)
            theta = -np.pi/2

        return np.array([x, y, theta], float).reshape(-1, 1)
