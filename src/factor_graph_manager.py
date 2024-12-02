import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple


class FactorGraphManager:
    """
    Class to construct the A matrix from a set of given odometry and sensor measurements.
    """
    def __init__(self,
                 inverse_odometry_model: Callable,
                 sensor_model: Callable,
                 initial_state: NDArray) -> None:
        """
        Initialize the FactorGraphManager.

        Parameters:
        inverse_odometry_model: Callable function of the inverse odometry model we are using. Input
            should be x0 and x1, output should be u.
        sensor_model: Callable function of the sensor model we are using. Input should be x and
            landmark ID, output should be z.
        """
        assert initial_state.shape == (3, 1)
        assert initial_state.dtype == np.float64

        self.inverse_odometry_model = inverse_odometry_model
        self.sensor_model = sensor_model
        self.initial_state = initial_state.copy()
        self.dim_state = initial_state.shape[0]
        self.landmarkIDs = []
        self.num_landmarks = 0
        self.num_measurements = 0
        self.poseID = 0
        self.odometry_info = []
        self.sensor_info = []

        # These match the values in simulator.py
        self._range_std = 0.1
        self._bearing_std = 0.1

        self.measurement_cov = np.array([[self._range_std**2, 0],
                                         [0, self._bearing_std**2]])
        
        # since diagonal we can just make the sqrt of the inverse be the sqrt of the diagonal
        self.sqrt_inv_measurement_cov = np.sqrt(self.measurement_cov)
        self.sqrt_inv_measurement_cov[self.sqrt_inv_measurement_cov != 0] = 1/self.sqrt_inv_measurement_cov[self.sqrt_inv_measurement_cov != 0]

        self._odometry_rotation_std = 0.05
        self._odometry_translation_std = 0.02

        self.odometry_cov = np.array([[self._odometry_rotation_std**2, 0, 0],
                                        [0, self._odometry_translation_std**2, 0],
                                        [0, 0, self._odometry_rotation_std**2]])
        
        # ditto for the odometry cov
        self.sqrt_inv_odometry_cov = np.sqrt(self.odometry_cov)
        self.sqrt_inv_odometry_cov[self.sqrt_inv_odometry_cov != 0] = 1/self.sqrt_inv_odometry_cov[self.sqrt_inv_odometry_cov != 0]


    def add_measurement(self, z: NDArray, H: Callable, J: Callable) -> None:
        """
        Adds a measurement to the factor graph.

        TODO: Currently requires that this is called after add_odometry, an error will occur
            otherwise.

        Parameters:
        z: The measurement to add to the factor graph. A 3x1 float numpy array [[range, bearing,
        ID]].T.
        """
        assert z.shape == (3, 1)
        assert z.dtype == np.float64
        # (landmarkID, poseID, measurementID, H, J, [range, bearing])
        landmarkID = int(z[2,0])
        info = (landmarkID, self.poseID, self.num_measurements, H, J, z[:2])
        self.sensor_info.append(info)
        self.num_measurements += 1
        if landmarkID not in self.landmarkIDs:
            self.num_landmarks += 1
            self.landmarkIDs.append(landmarkID)

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
        assert x.dtype == np.float64
        assert x.shape[0] == self.dim_state*(self.poseID + 1) + 2*self.num_landmarks

        A = np.zeros((self.dim_state*self.poseID + self.dim_state + 2*self.num_measurements, len(x)))
        b = np.zeros((self.dim_state*self.poseID + self.dim_state + 2*self.num_measurements, 1))

        # Set the prior
        A[:self.dim_state,:self.dim_state] = -np.eye(self.dim_state) * 1000  # Make the prior very confident
        b[:self.dim_state] = self.sqrt_inv_measurement_cov @ (x[:self.dim_state] - self.initial_state)

        for odometry_data in self.odometry_info:
            poseID, u, F, G = odometry_data
            poseID += 1
            current = poseID*self.dim_state
            previous = current - self.dim_state
            next = current + self.dim_state
            F_evaluated = F(x[previous:current], u).reshape(self.dim_state, self.dim_state)
            G_evaluated = G(x[current:next])
            b[current:next] = self.sqrt_inv_odometry_cov @ (u - self.inverse_odometry_model(x[previous:current], x[current:next]))
            A[current:next, previous:current] = self.sqrt_inv_odometry_cov @ F_evaluated
            A[current:next, current:next] = self.sqrt_inv_odometry_cov @ G_evaluated

        for sensor_data in self.sensor_info:
            landmarkID, measured_poseID, measurementID, H, J, z = sensor_data
            landmark_start = (1+self.poseID)*self.dim_state
            pose_current = measured_poseID*self.dim_state
            pose_next = pose_current + self.dim_state
            landmark_current = landmark_start + 2*landmarkID
            landmark_next = landmark_current + 2
            measurement_current = (1+self.poseID)*self.dim_state + 2*measurementID
            measurement_next = measurement_current + 2
            H_evaluated = H(x[pose_current:pose_next], x[landmark_current:landmark_next]).reshape(2, 3)
            J_evaluated = J(x[pose_current:pose_next], x[landmark_current:landmark_next]).reshape(len(z), len(z))
            b[measurement_current:measurement_next] = self.sqrt_inv_measurement_cov @ (z - self.sensor_model(x[pose_current:pose_next], x[landmark_current:landmark_next]))
            A[measurement_current:measurement_next, pose_current:pose_next] = self.sqrt_inv_measurement_cov @ H_evaluated
            A[measurement_current:measurement_next, landmark_current:landmark_next] = self.sqrt_inv_measurement_cov @ J_evaluated

        return A, b

