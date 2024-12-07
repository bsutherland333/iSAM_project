#!/usr/bin/env python3

import argparse
import numpy as np

import matplotlib.pyplot as plt

from data_parser import DataParser
from factor_graph_manager import FactorGraphManager
from models import motion_model, inverse_motion_model, sensor_model, inverse_sensor_model
from plotter import plot_factor_graph
from simulator import Simulator
from jax_jacobians import make_F_H_J_G

# Change what is imported to change the methods
from column_reorderers import passthrough as reorder
from iterative_matrix_augmenters import update_measurement as augment_r_measurement
from iterative_matrix_augmenters import update_variable as augment_r_variable
from qr_factorizers import np_qr as qr
from solvers import scipy_solver as solver

from models import _wrap_within_pi

# Set numpy to not wrap arrays
np.set_printoptions(linewidth=np.inf, suppress=False)

def main(num_iterations: int, data_filepath: str, use_iterative_solver: bool, num_iters_before_batch: int, plot_live: bool):
    range_measurement_std = 0.1
    bearing_measurement_std = 0.05
    odom_translation_std = 0.1
    odom_rotation_std = 0.05
    if data_filepath != '':
        data = DataParser(data_filepath)
        x = data.get_initial_state()
        true_landmark_positions = data.get_landmark_positions()
    else:
        x = np.array([0.5, 0.5, 0], float).reshape(-1,1)
        true_landmark_positions = np.array([[0, 0], [3, 0], [6, 0], [0, 3], [3, 3], [6, 3]], float).T
        data = Simulator(inverse_motion_model,
                         sensor_model,
                         np_seed=2,
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

    # Variables kept for a history and to XL order the variables before batch updating
    pose_best_estimate = x.copy()
    landmark_best_estimate = np.array([[]]).T

    # Housekeeping
    landmark_id_hist = []

    # Variables used by the incremental factorizer. Create the initial R matrix.
    A, b = factor_graph_manager.get_A_b_matrix(x)
    Q, R = np.linalg.qr(A)
    P = None    # Create a variable so it stays in scope
    d = Q.T @ b
    x = x.copy()
    do_find_odometry = False    # Flag to know if we need to find the odometry in the state or if it is the last one
    len_x_at_last_reorder = 0
    len_pose_hist_at_last_reorder = len(pose_best_estimate)
    added_pose_hist_idx_in_x = []   # Keeps track of the indices of the pose hists that were added in the incremental factorizer
    added_landmark_hist_idx_in_x = []   # Keeps track of the indices of the landmark hists that were added in the incremental factorizer
    #idxs_of_thetas = np.zeros_like(x, int)     # Keeps track of the indices of the theta values (so we can wrap them)
    #idxs_of_thetas[2][0] = 1
    last_odom_idx = 0   # Tracker to indicate where in the state array the last odometry is kept

    for timestep in range(num_iterations):
        if not use_iterative_solver or timestep % num_iters_before_batch == 0:
            perform_batch_update = True
        else:
            perform_batch_update = False

        # Run the simulator/data parser
        u, z, x_truth = data.get_next_timestep()
        x_truth_hist.append(x_truth)
        measurement_hist.extend([np.array([timestep + 1, z[2, i]], int).reshape(-1,1) for i in range(z.shape[1])])

        #################
        #    Odometry   #
        #################
        # Get the next pose based on the odometry.
        # Use the current linearization point for incremental factorization, and the best linearization
        # point for batch updates.
        previous_state = pose_best_estimate[-3:].copy() if perform_batch_update else x[-3:].copy()
        current_state = motion_model(previous_state, u)

        # TODO: Remove this line after debugging
        u = inverse_motion_model(x_truth_hist[-2], x_truth_hist[-1])

        # Add the odometry to the factor graph manager for batch updates.
        # Also append new pose to the pose history.
        factor_graph_manager.add_odometry(u, F, G)
        pose_best_estimate = np.vstack((pose_best_estimate, current_state))

        # [INCREMENTAL FACTORIZER] Add the new pose to the state vector and the R matrix
        # Compute the F and G Jacobians
        F_eval = factor_graph_manager.sqrt_inv_odometry_cov @ F(previous_state, current_state).reshape(3,3)
        G_eval = factor_graph_manager.sqrt_inv_odometry_cov @ G(previous_state, current_state).reshape(3,3)

        # Form the w vector by inserting F in the location it would be in the XL ordered vector
        w = np.zeros((R.shape[1], F_eval.shape[0]))
        w[last_odom_idx:last_odom_idx+3,:] = F_eval.T

        if do_find_odometry:
            # Reorder w based on P
            w = P @ w
            do_find_odometry = False

        # Transpose and append G
        w_T = np.hstack((w.T, G_eval))

        # Compute gamma (the RHS of Rx=d)
        d_prime = np.array(u - inverse_motion_model(previous_state, current_state))
        gamma = factor_graph_manager.sqrt_inv_odometry_cov @ d_prime
        # Augment R and d with w_T and gamma
        R, d = augment_r_variable(len(current_state), w_T, gamma, R, d)
        # Add variable to the state vector
        x = np.vstack((x, current_state))
        last_odom_idx = len(x)-3    # Keep track of this in case we add a new landmark to the state
        added_pose_hist_idx_in_x.append(last_odom_idx)  # So we know how to reorder the poses into pose hist

        ######################
        ##    Measurements   #
        ######################
        #for i in range(z.shape[1]):
        #    current_measurement = z[:, i].reshape(-1, 1)
        #    factor_graph_manager.add_measurement(current_measurement, H, J)
        #    landmark_id = int(current_measurement[2][0])

        #    do_add_landmark_to_state = False

        #    if landmark_id not in landmark_id_hist:
        #        landmark_guess = inverse_sensor_model(current_state, current_measurement[:2])
        #        landmark_best_estimate = np.vstack((landmark_best_estimate, landmark_guess))
        #        landmark_id_hist.append(landmark_id)
        #        do_add_landmark_to_state = True

        #    # [INCREMENTAL FACTORIZER] Add the new measurement to the R vector
        #    # Compute the H and J Jacobians
        #    current_landmark_state = landmark_best_estimate[landmark_id*2:landmark_id*2+2]
        #    H_eval = factor_graph_manager.sqrt_inv_measurement_cov @ H(current_state, current_landmark_state).reshape(2,3)
        #    J_eval = factor_graph_manager.sqrt_inv_measurement_cov @ J(current_state, current_landmark_state).reshape(2,2)

        #    # Form the w vector
        #    w = np.zeros((R.shape[1], H_eval.shape[0]))
        #    w[-3:,:] = H_eval.T  # The Jacobian wrt the state will always take the last added state 
        #                         # (the current state -- which is [-3:] in the state vector since we just appended it)

        #    # Compute gamma
        #    d_prime = np.array(current_measurement[:2] - sensor_model(current_state, current_landmark_state))
        #    gamma = factor_graph_manager.sqrt_inv_measurement_cov @ d_prime

        #    # If the landmark is new, just add it to the state vector
        #    if do_add_landmark_to_state:
        #        # Add the J to the end since we are adding a new landmark to the state
        #        w_T = np.hstack((w.T, J_eval))

        #        R, d = augment_r_variable(2, w_T, gamma, R, d)
        #        x = np.vstack((x, current_landmark_state))
        #        # Note: Don't update the current pose idx
        #        added_landmark_hist_idx_in_x.append(len(x)-2)
        #    else:
        #        # Add J where it should be based on the XL ordered landmark
        #        idx = len(pose_best_estimate) + landmark_id*2
        #        w[idx:idx+2,:] = J_eval.T

        #        # Reorder only the part of w that corresponds to the length of x at the last batch update
        #        # (that is the only part of x that was reordered)
        #        w[:len_x_at_last_reorder,:] = P @ w[:len_x_at_last_reorder,:]

        #        R, d = augment_r_measurement(w.T, gamma, R, d)

        ###############
        #    Solve!   #
        ###############
        if perform_batch_update:
            # Stack the x vector for solving 
            x = np.vstack((pose_best_estimate, landmark_best_estimate))

            # Solve the linear system with a batch update
            A, b = factor_graph_manager.get_A_b_matrix(x)
            A_prime, P = reorder(A)

            # Factor A
            Q, R = qr(A_prime)
            d = Q.T @ b

            x_prime = solver(R, d)
            x_corrected = x + P @ x_prime

            # wrap the theta portions of the state
            # TODO: do we keep this? it is currently doing bad things
            #for i in range(len(pose_hist)//3):
            #    x[i*3-1][0] = _wrap_within_pi(x[i*3-1][0])

            # Set the flag to tell incremental solver that we need to look for the odometry in the state vector
            do_find_odometry = True
            last_odom_idx = len(pose_best_estimate) - 3
            len_x_at_last_reorder = len(x)
            len_pose_hist_at_last_reorder = len(pose_best_estimate)
            added_pose_hist_idx_in_x = []
            added_landmark_hist_idx_in_x = []

            # Unpack the x vector into the pose_hist and landmark_hist arrays
            pose_best_estimate = x_corrected[:len(pose_best_estimate)]
            landmark_best_estimate = x_corrected[len(pose_best_estimate):]

        else:
            # Update the factorization incremental and solve
            # Updating was already done. Just solve!
            x_delta_prime = solver(R, d)
            x_corrected = x + x_delta_prime

            # Unpack the x vector into the pose_hist and landmark_hist arrays
            # First, unorder the part that got ordered
            reordered_x = P @ x_corrected[:len_x_at_last_reorder]

            # Get the indices of the poses and landmarks that were added in the incremental factorization step
            pose_hist_slice_indices = np.array([[i, i+1, i+2] for i in added_pose_hist_idx_in_x]).flatten()
            landmark_hist_slice_indices = np.array([[i, i+1] for i in added_landmark_hist_idx_in_x]).flatten()

            # Extract the landmarks and poses and save to the hist variables
            pose_best_estimate = np.vstack((reordered_x[:len_pose_hist_at_last_reorder], x_corrected[pose_hist_slice_indices]))
            if len(landmark_hist_slice_indices) < 1:
                landmark_best_estimate = reordered_x[len_pose_hist_at_last_reorder:len_x_at_last_reorder]
            else:
                landmark_best_estimate = np.vstack((reordered_x[len_pose_hist_at_last_reorder:len_x_at_last_reorder], x_corrected[landmark_hist_slice_indices]))


        for i,val in enumerate(pose_hist):
            if i % 3 - 1 == 0:
                assert abs(val[0]) < np.pi
        
        if plot_live or timestep == num_iterations - 1:
            plot_factor_graph(estimated_robot_poses=pose_best_estimate.reshape(-1, 3).T,
                              true_robot_poses=np.hstack(x_truth_hist),
                              estimated_landmark_positions=landmark_best_estimate.reshape(-1, 2).T,
                              true_landmark_positions=true_landmark_positions,
                              #measurement_associations=np.hstack(measurement_hist),
                              hold_on=True if timestep == num_iterations - 1 else False
                              )

        plt.waitforbuttonpress()

        print(f"timestep: {timestep}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-iterations', type=int, default=200, help='Number of iterations to run the algorithm for.')
    parser.add_argument('-f', '--data-filepath', type=str, default='', help='Filepath to recorded data. If not provided, will use simulated data.')
    parser.add_argument('--use-iterative-solver', type=bool, default=False, help='Use iterative solver (iSAM) instead of batch solver (SAM).')
    parser.add_argument('--num-iters-before-batch', type=int, default=25, help='Number of iterations before running batch solver.')
    parser.add_argument('-p', '--plot-live', type=bool, default=True, help='Plot the factor graph at each timestep.')
    args = vars(parser.parse_args())

    main(**args)
