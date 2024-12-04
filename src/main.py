#!/usr/bin/env python3

import argparse
import numpy as np

from data_parser import DataParser
from factor_graph_manager import FactorGraphManager
from models import motion_model, inverse_motion_model, sensor_model, inverse_sensor_model
from plotter import plot_factor_graph
from simulator import Simulator
from jax_jacobians import make_F_H_J_G

# Change what is imported to change the methods
from column_reorderers import colamd as reorder
from iterative_matrix_augmenters import update_measurement as augment_r_measurement
from iterative_matrix_augmenters import update_variable as augment_r_variable
from qr_factorizers import np_qr as qr
from solvers import scipy_solver as solver

# Set numpy to not wrap arrays
np.set_printoptions(linewidth=np.inf)

def main(num_iterations: int, data_filepath: str, use_iterative_solver: bool, num_iters_before_batch: int, plot_live: bool):
    range_measurement_std = 0.1
    bearing_measurement_std = 0.05
    odom_translation_std = 0.05
    odom_rotation_std = 0.02
    if data_filepath != '':
        data = DataParser(data_filepath)
        x = data.get_initial_state()
        true_landmark_positions = data.get_landmark_positions()
    else:
        x = np.array([0.5, 0.5, 0], float).reshape(-1,1)
        true_landmark_positions = np.array([[0, 0], [3, 0], [6, 0], [0, 3], [3, 3], [6, 3]], float).T
        data = Simulator(inverse_motion_model,
                         sensor_model,
                         np_seed=0,
                         initial_state=x,
                         landmark_locations=true_landmark_positions,
                         range_measurement_std=range_measurement_std,
                         bearing_measurement_std=bearing_measurement_std,
                         odometry_translation_std=odom_translation_std,
                         odometry_rotation_std=odom_rotation_std)

    F, H, J, G = make_F_H_J_G(inverse_motion_model, sensor_model)
    factor_graph_manager = FactorGraphManager(inverse_motion_model,
                                              sensor_model,
                                              x,
                                              range_measurement_std,
                                              bearing_measurement_std,
                                              odom_rotation_std,
                                              odom_translation_std,
                                              )

    x_truth_hist = [x]
    measurement_hist = []
    pose_hist = x.copy()
    landmark_hist = np.array([[]]).T
    landmark_id_hist = []
    R = None
    for timestep in range(num_iterations):
        # Run the simulator/data parser
        u, z, x_truth = data.get_next_timestep()
        x_truth_hist.append(x_truth)
        measurement_hist.extend([np.array([timestep + 1, z[2, i]], int).reshape(-1,1) for i in range(z.shape[1])])

        # Add the measurements to the factor graph
        # Also append states to the state histories
        previous_state = pose_hist[-3:]
        factor_graph_manager.add_odometry(u, F, G)
        pose_hist = np.vstack((pose_hist, motion_model(previous_state, u)))

        current_state = pose_hist[-3:]
        for i in range(z.shape[1]):
            current_measurement = z[:, i].reshape(-1, 1)
            factor_graph_manager.add_measurement(current_measurement, H, J)

            if z[2,i] not in landmark_id_hist:
                landmark_guess = inverse_sensor_model(current_state, current_measurement[:2])
                landmark_hist = np.vstack((landmark_hist, landmark_guess))
                landmark_id_hist.append(z[2, i].astype(int))

        # Stack the x vector for solving 
        x = np.vstack((pose_hist, landmark_hist))
        if not use_iterative_solver or timestep % num_iters_before_batch == 0:
            # Solve the linear system with a batch update
            A, b = factor_graph_manager.get_A_b_matrix(x)
            A_prime, P = reorder(A)

            # Factor A
            Q, R = qr(A_prime)
            d = Q.T @ b

            x_prime = solver(R, d)
            x += P @ x_prime

        else:
            # Update the factorization iteratively and solve

            # First update with the odometry
            # ASSUMPTION: We receive exactly 1 odometry measurment every timestep (i.e. we add 1 new pose variable)
            sqrt_odom_cov = factor_graph_manager.sqrt_inv_odometry_cov
            gamma = sqrt_odom_cov @ (u - inverse_motion_model(previous_state, current_state))
            F_evaluated = F(previous_state, u).reshape(3, 3)
            G_evaluated = G(current_state)
            w_T = np.zeros((F_evaluated.shape[0], R.shape[1]))
            w_T[:,-F_evaluated.shape[1]:] = sqrt_odom_cov @ F_evaluated
            w_T = np.hstack((w_T, sqrt_odom_cov @ G_evaluated))
            R, d = augment_r_variable(3, w_T, gamma, R, d)
            
            for i in range(z.shape[1]):
                break
                current_measurement = z[:, i].reshape(-1, 1)
                landmark_guess = inverse_sensor_model(current_state, current_measurement[:2])

                sqrt_meas_cov = factor_graph_manager.sqrt_inv_measurement_cov
                gamma = sqrt_meas_cov @ (z[:2] - self.sensor_model(current_state, landmark_guess))

                # TODO: Finish this...
                #if z[2,i] not in landmark_id_hist:
                #    H_evaluated = H(x[pose_current:pose_next], x[landmark_current:landmark_next]).reshape(2, 3)
                #    J_evaluated = J(x[pose_current:pose_next], x[landmark_current:landmark_next]).reshape(len(z), len(z))
                #    w_T = np.zeros(())
                #    w_T[] = sqrt_meas_cov @ H_evaluated
                #    w_T = np.hstack((w_T, sqrt_meas_cov @ J_evaluated))

                #    R, d = augment_r_variable(2, w_T, gamma, R, d)
                #else:

                #    R, d = augment_r_measurement(w_T, gamma, R, d)
                
            x_delta_prime = solver(R, d)
            #x += x_delta_prime

        # Unpack the x vector into the pose_hist and landmark_hist arrays
        pose_hist = x[:len(pose_hist)]
        landmark_hist = x[len(pose_hist):]

        if plot_live or timestep == num_iterations - 1:
            plot_factor_graph(estimated_robot_poses=pose_hist.reshape(-1, 3).T,
                              true_robot_poses=np.hstack(x_truth_hist),
                              estimated_landmark_positions=landmark_hist.reshape(-1, 2).T,
                              true_landmark_positions=true_landmark_positions,
                              #measurement_associations=np.hstack(measurement_hist),
                              hold_on=True if timestep == num_iterations - 1 else False
                              )

        print(f"timestep: {timestep}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-iterations', type=int, default=200, help='Number of iterations to run the algorithm for.')
    parser.add_argument('-f', '--data-filepath', type=str, default='', help='Filepath to recorded data. If not provided, will use simulated data.')
    parser.add_argument('--use-iterative-solver', type=bool, default=False, help='Use iterative solver (iSAM) instead of batch solver (SAM).')
    parser.add_argument('--num-iters-before-batch', type=int, default=25, help='Number of iterations before running batch solver.')
    parser.add_argument('-p', '--plot-live', type=bool, default=False, help='Plot the factor graph at each timestep.')
    args = vars(parser.parse_args())

    main(**args)
