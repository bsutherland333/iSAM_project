import numpy as np

from src.covariance_extraction import get_robot_and_landmark_covariances


# Set up test data
num_robot_poses = 12
num_landmarks = 8
num_factors = num_robot_poses*3 + 2 + num_landmarks*2*3
A = np.random.normal(loc=0, scale=4, size=(num_factors, num_robot_poses*3 + num_landmarks*2))

R = np.linalg.qr(A, mode='r')

def test_get_robot_and_landmark_covariances():
    print(R.shape)
    robot_covs, landmark_covs = get_robot_and_landmark_covariances(num_robot_poses*3, R)

    assert robot_covs.shape == (num_robot_poses, 3, 3)
    assert landmark_covs.shape == (num_landmarks, 2, 2)

    assert list(np.flatnonzero(robot_covs))
    assert list(np.flatnonzero(landmark_covs))

