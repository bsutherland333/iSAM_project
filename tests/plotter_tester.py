# Disgusting code for testing the plotter

import numpy as np
import matplotlib.pyplot as plt

from src.simulator import Simulator
from src.models import sensor_model, inverse_motion_model, motion_model

from src.plotter import plot_factor_graph

sim = Simulator(inverse_motion_model, sensor_model, dt=0.5)
x = sim._initial_state

x_true_hist = [x.flatten()]
x_est_hist = [x.flatten()]

measurement_associations = []
for i in range(50):
    u, z, x_true = sim.get_next_timestep()

    x = motion_model(x, u)
    x_true_hist.append(x_true.flatten())
    x_est_hist.append(x.flatten())

    x_true_array = np.array(x_true_hist).T
    x_est_array = np.array(x_est_hist).T

    landmark_array = []
    for landmark in sim._landmarks.T:
        landmark = landmark.copy()
        landmark = sim._R_sim_truth @ landmark
        landmark += sim._initial_state[0:2].flatten()
        landmark_array.append(landmark)
    landmark_array = np.array(landmark_array).T
    est_landmark_array = landmark_array + np.random.normal(0, 0.1, landmark_array.shape)

    landmark_sorted_indices = np.argsort(sim._landmark_ids[np.where(sim._landmark_ids != -1)])
    est_landmark_array = est_landmark_array[:, np.where(sim._landmark_ids != -1)].squeeze()
    est_landmark_array = est_landmark_array[:, landmark_sorted_indices]

    landmark_covariances = []
    for j in range(est_landmark_array.shape[1]):
        covariance = np.array([[0.5**2, 0.1], [0.1, 0.3**2]])
        landmark_covariances.append(covariance)
    landmark_covariances = np.array(landmark_covariances)

    robot_covariances = []
    for j in range(x_est_array.shape[1]):
        covariance = np.array([[0.2**2, 0, 0], [0, 0.1**2, 0], [0, 0, 0.1**2]])
        robot_covariances.append(covariance)
    robot_covariances = np.array(robot_covariances)

    for j in range(z.shape[1]):
        measurement_associations.append([i + 1, z[2, j]])
    measurement_associations_array = np.array(measurement_associations, int).T

    plot_factor_graph(true_robot_poses=x_true_array,
               estimated_robot_poses=x_est_array,
               estimated_robot_covariances=robot_covariances,
               true_landmark_positions=landmark_array,
               estimated_landmark_positions=est_landmark_array,
               estimated_landmark_covariances=landmark_covariances,
               measurement_associations=measurement_associations_array,
               pause_duration=1,
               )

plt.show()
