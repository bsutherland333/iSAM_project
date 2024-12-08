import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_factor_graph(estimated_robot_poses=None,
                      estimated_robot_covariances=None,
                      true_robot_poses=None,
                      estimated_landmark_positions=None,
                      estimated_landmark_covariances=None,
                      true_landmark_positions=None,
                      measurement_associations=None,
                      pause_duration=0.4,
                      hold_on=False,
                      plot_heading_vector=False,
                      ):
    """
    Plots the robot, landmarks, and measurements (estimated and true) in a 2D plot. All inputs are
    optional, allowing use with whatever data is avaliable.

    Parameters:
    estimated_robot_poses: (3, n) numpy array of estimated robot poses.
    estimated_robot_covariances: (n, 3, 3) numpy array of estimated robot pose covariances.
    true_robot_poses: (3, n) numpy array of true robot poses.
    estimated_landmark_positions: (2, m) numpy array of estimated landmark positions.
    estimated_landmark_covariances: (m, 2, 2) numpy array of estimated landmark position covariances.
    true_landmark_positions: (2, m) numpy array of true landmark positions.
    measurement_associations: (2, k) numpy array of measurement associations, where each column
        is a robot pose and landmark pair. The first row is the robot pose index, based on the
        order of the estimated_robot_poses array, and the second row is the landmark index, based
        on the order of estimated_landmark_positions.
    pause_duration: The amount of time to pause after plotting, in seconds.
    hold_on: If True, the plot will pause execution until the window is closed.
    """
    assert estimated_robot_poses is None or estimated_robot_poses.ndim == 2
    assert estimated_robot_poses is None or estimated_robot_poses.shape[0] == 3
    assert estimated_robot_poses is None or estimated_robot_poses.dtype == np.float64

    assert estimated_robot_covariances is None or estimated_robot_covariances.ndim == 3
    assert estimated_robot_covariances is None or estimated_robot_poses is None or \
        estimated_robot_covariances.shape[0] == estimated_robot_poses.shape[1]
    assert estimated_robot_covariances is None or estimated_robot_covariances.shape[1] == 3
    assert estimated_robot_covariances is None or estimated_robot_covariances.shape[2] == 3
    assert estimated_robot_covariances is None or estimated_robot_covariances.dtype == np.float64

    assert true_robot_poses is None or true_robot_poses.ndim == 2
    assert true_robot_poses is None or true_robot_poses.shape[0] == 3
    assert true_robot_poses is None or true_robot_poses.dtype == np.float64

    assert estimated_landmark_positions is None or estimated_landmark_positions.ndim == 2
    assert estimated_landmark_positions is None or estimated_landmark_positions.shape[0] == 2
    assert estimated_landmark_positions is None or estimated_landmark_positions.dtype == np.float64

    assert estimated_landmark_covariances is None or estimated_landmark_covariances.ndim == 3
    assert estimated_landmark_covariances is None or estimated_landmark_positions is None or \
        estimated_landmark_covariances.shape[0] == estimated_landmark_positions.shape[1]
    assert estimated_landmark_covariances is None or estimated_landmark_covariances.shape[1] == 2
    assert estimated_landmark_covariances is None or estimated_landmark_covariances.shape[2] == 2
    assert estimated_landmark_covariances is None or estimated_landmark_covariances.dtype == np.float64

    assert true_landmark_positions is None or true_landmark_positions.ndim == 2
    assert true_landmark_positions is None or true_landmark_positions.shape[0] == 2
    assert true_landmark_positions is None or true_landmark_positions.dtype == np.float64

    assert measurement_associations is None or measurement_associations.ndim == 2
    assert measurement_associations is None or measurement_associations.shape[0] == 2
    assert measurement_associations is None or measurement_associations.dtype == np.int64
    assert measurement_associations is None or estimated_robot_poses is None or \
        np.max(measurement_associations[0, :]) == estimated_robot_poses.shape[1] - 1
    assert measurement_associations is None or estimated_landmark_positions is None or \
        np.max(measurement_associations[1, :]) == estimated_landmark_positions.shape[1] - 1

    assert pause_duration >= 0

    plt.clf()

    # Plot measurements
    if measurement_associations is not None and estimated_robot_poses is not None and \
            estimated_landmark_positions is not None:
        for i in range(measurement_associations.shape[1]):
            robot_pose = estimated_robot_poses[:, measurement_associations[0, i]]
            landmark_pose = estimated_landmark_positions[:, measurement_associations[1, i]]
            plt.plot([robot_pose[0], landmark_pose[0]], [robot_pose[1], landmark_pose[1]], 'k-', linewidth=1)

    # Plot true landmark positions
    if true_landmark_positions is not None:
        plt.plot(true_landmark_positions[0, :], true_landmark_positions[1, :], 'go', label='True Landmark Positions')

    # Plot estimated landmark positions
    if estimated_landmark_positions is not None:
        plt.plot(estimated_landmark_positions[0, :], estimated_landmark_positions[1, :], 'g+', label='Estimated Landmark Positions', markersize=10)

    # Plot estimated landmark covariances with 3 sigma ellipses
    if estimated_landmark_positions is not None and estimated_landmark_covariances is not None:
        for i in range(estimated_landmark_covariances.shape[0]):
            cov = estimated_landmark_covariances[i, :, :]
            eigvals, eigvecs = np.linalg.eig(cov)
            angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
            ell = Ellipse(xy=(estimated_landmark_positions[0, i], estimated_landmark_positions[1, i]),
                          width=3 * np.sqrt(eigvals[0]),
                          height=3 * np.sqrt(eigvals[1]),
                          angle=np.rad2deg(angle),
                          edgecolor='k',
                          facecolor='none',
                          zorder=4)
            plt.gca().add_artist(ell)

    # Plot true robot poses
    if true_robot_poses is not None:
        plt.plot(true_robot_poses[0, :], true_robot_poses[1, :], 'bo-', label='True Robot Poses')

    # Plot estimated robot poses
    if estimated_robot_poses is not None:
        plt.plot(estimated_robot_poses[0, :], estimated_robot_poses[1, :], 'ro-', label='Estimated Robot Poses')
        if plot_heading_vector:
            plt.quiver(estimated_robot_poses[0, :], estimated_robot_poses[1, :], np.cos(estimated_robot_poses[2,:]), np.sin(estimated_robot_poses[2,:]), zorder=3, width=2, units='dots')

    # Plot estimated robot pose covariances with 3 sigma ellipses
    if estimated_robot_poses is not None and estimated_robot_covariances is not None:
        for i in range(estimated_robot_covariances.shape[0]):
            cov = estimated_robot_covariances[i, :2, :2]
            eigvals, eigvecs = np.linalg.eig(cov)
            angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
            ell = Ellipse(xy=(estimated_robot_poses[0, i], estimated_robot_poses[1, i]),
                          width=3 * np.sqrt(eigvals[0]),
                          height=3 * np.sqrt(eigvals[1]),
                          angle=np.rad2deg(angle),
                          edgecolor='k',
                          facecolor='none',
                          zorder=4)
            plt.gca().add_artist(ell)

    plt.legend()
    plt.axis('equal')
    plt.pause(pause_duration)

    if hold_on:
        plt.show()

