import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple


class FactorGraphManager:
    """
    Class to construct the A matrix from a set of given odometry and sensor measurements.
    """
    def __init__(self, initial_state: NDArray, dim_state: int) -> None:
        """
        Initialize the FactorGraphManager.

        Parameters:
        odometry_model: Callable function of the odometry model we are using. Input should be x and u,
            output should be x.
        sensor_model: Callable function of the sensor model we are using. Input should be x and
            landmark ID, output should be z.
        """

        self.x = initial_state
        self.dim_state = dim_state
        self.prev_A_width = initial_state.shape[0]
        self.num_landmarks = 0
        self.poseID = 0
        self.odometry_info = []
        self.sensor_info = []

    def add_measurement(self, z: NDArray, H: Callable, J: Callable) -> None:
        """
        Adds a measurement to the factor graph.

        Parameters:
        z: The measurement to add to the factor graph. A 3x1 float numpy array [[range, bearing,
        ID]].T.
        """
        assert z.shape == (3, 1)
        assert z.dtype == np.float64
        # (landmarkID, poseID, H, J, [range, bearing])
        info = (z[2], self.poseID, H, J, z[:2])
        self.sensor_info.append(info)

    def add_odometry(self, u: NDArray, F: Callable, G: Callable) -> None:
        """
        Adds an odometry measurement to the factor graph.

        Parameters:
        u: The odometry measurement to add to the factor graph.
        """
        assert u.shape == (3, 1)
        assert u.dtype == np.float64
        # (poseID, u, F, G)
        info = (self.poseID, u, F, G)
        self.odometry_info.append(info)
        self.poseID += 1

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

        A = np.zeros((self.dim_state*self.poseID + 2*self.num_landmarks, len(x)))
        b = np.zeros((self.dim_state*self.poseID + 2*self.num_landmarks, 1))

        for odometry_data in self.odometry_info:
            poseID, u, F, G = odometry_data
            current = poseID*self.dim_state
            previous = (poseID-1)*self.dim_state
            next = (poseID+1)*self.dim_state
            F_evaluated = F(x[previous:current], u).reshape(self.dim_state, self.dim_state)
            G_evaluated = G(x[current:next])
            b[current:next] = u  # TODO: we have to pre-multiply by inverse transpose sqrt of process noise
            A[current:next, previous:current] = F_evaluated # TODO: we have to pre-multiply by inverse transpose sqrt of process noise
            A[current:next, current:next] = G_evaluated # TODO: we have to pre-multiply by inverse transpose sqrt of process noise

        for sensor_data in self.sensor_info:
            landmarkID, measured_poseID, H, J, z = sensor_data
            landmark_start = self.poseID*self.dim_state
            pose_current = measured_poseID*self.dim_state
            pose_next = pose_current + self.dim_state
            landmark_current = landmark_start + 2*landmarkID
            landmark_next = landmark_current + 2
            H_evaluated = H(x[pose_current:pose_next], x[landmark_current:landmark_next]).reshape(len(z), self.dim_state)
            J_evaluated = J(x[pose_current:pose_next], x[landmark_current:landmark_next]).reshape(len(z), len(z))
            b[landmark_current:landmark_next] = z # TODO: we have to pre-multiply by inverse transpose sqrt of measurement noise
            A[landmark_current:landmark_next, pose_current:pose_next] = H_evaluated # TODO: we have to pre-multiply by inverse transpose sqrt of measurement noise
            A[landmark_current:landmark_next, landmark_current:landmark_next] = J_evaluated # TODO: we have to pre-multiply by inverse transpose sqrt of measurement noise


        return A, b

