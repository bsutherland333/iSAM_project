import numpy as np
from numpy.typing import NDArray

def naive_covariance_extractor(R: NDArray) -> NDArray:
    '''
    Extracts the covariance of each landmark from the Cholesky triangle.

    Parameters:
    R: (n,n) Square Cholesky triangle.

    Returns:
    M: (n,n) Square covariance matrix
    '''
    assert R.shape[0] == R.shape[1]

    M = np.linalg.inv(R.T @ R)

    return M

def get_robot_and_landmark_covariances(index: int, R: NDArray) -> tuple[NDArray, NDArray]:
    

    assert index <= R.shape[0]
    assert R.shape[0] == R.shape[1]

    M = naive_covariance_extractor(R)

    robot_covariances = np.zeros((index//3, 3, 3))
    landmark_covariances = np.zeros(((R.shape[0] - index)//2, 2, 2))

    for i in range(index//3):
        robot_covariances[i] = M[i*3:i*3+3,i*3:i*3+3]

    for i in range((R.shape[0]-index)//2):
        landmark_covariances[i] = M[index+i*2:index+i*2+2, index+i*2:index+i*2+2]

    return robot_covariances, landmark_covariances



