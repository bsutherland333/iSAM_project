#!/usr/bin/env python3

import argparse
import numpy as np

from data_parser import DataParser
from factor_graph_manager import FactorGraphManager
from models import motion_model, inverse_motion_model, sensor_model
from plotter import plot_factor_graph
from simulator import Simulator
from jax_jacobians import make_F_H_J_G

# Change what is imported to change the methods
from column_reorderers import colamd as reorder
from iterative_matrix_augmenters import augment_r as augment_r
from qr_factorizers import np_qr as qr
from solvers import scipy_solver as solver


def main(num_iterations: int, data_filepath: str, use_iterative_solver: bool, num_iters_before_batch: int, plot_live: bool):
    if data_filepath != '':
        data = DataParser(data_filepath)
        x = data.get_initial_state()
    else:
        data = Simulator(inverse_motion_model, sensor_model, np_seed=0)
        x = data._initial_state

    F, H, J, G = make_F_H_J_G(motion_model, sensor_model)
    factor_graph_manager = FactorGraphManager(x, 3)

    x_truth_hist = []
    measurement_hist = []
    for timestep in range(num_iterations):
        # Run the simulator/data parser
        u, z, x_truth = data.get_next_timestep()
        x_truth_hist.append(x_truth)
        measurement_hist.append([timestep + 1, z[2, i]] for i in range(z.shape[1]))

        # Add the measurements to the factor graph and get the A matrix
        for i in range(z.shape[1]):
            factor_graph_manager.add_measurement(z[:, i].reshape(-1, 1), H, J)
        factor_graph_manager.add_odometry(u, F, G)

        if not use_iterative_solver or timestep % num_iters_before_batch == 0:
            # Solve the linear system with a batch update
            A, b = factor_graph_manager.get_A_b_matrix(x)
            A_prime, P = reorder(A)
            Q, R = qr(A_prime)
            x_prime = solver(R, Q.T @ b)
            x = P @ x_prime
        else:
            # Solve the linear system iteratively
            # TODO: I don't really know how this works yet, so this is probably wrong
            for i in range(z.shape[1]):
                R = augment_r(R, z[:, i].reshape(-1, 1))
            x_prime = solver(R, Q.T @ b)
            x = P @ x_prime

        if plot_live or timestep == num_iterations - 1:
            # TODO: Untested reformatting/plotting code, make sure data looks correct before
            # attempting to plot
            if data_filepath == '':
                true_landmark_positions = data._landmarks.copy()
                true_landmark_positions = data._R_sim_truth @ true_landmark_positions
                true_landmark_positions += data._initial_state[0:2].flatten()
            else:
                true_landmark_positions = None

            plot_factor_graph(estimated_robot_poses=x[:(timestep + 1)*3].reshape(3, -1),
                              true_robot_poses=np.hstack(x_truth_hist),
                              estimated_landmark_positions=x[(timestep + 1)*3:].reshape(2, -1),
                              true_landmark_positions=true_landmark_positions,
                              measurement_associations=np.array(measurement_hist, int).T,
                              hold_on=True if timestep == num_iterations - 1 else False
                              )

        print(f"timestep: {timestep}, x: {x.flatten()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-iterations', type=int, default=200, help='Number of iterations to run the algorithm for.')
    parser.add_argument('-f', '--data-filepath', type=str, default='', help='Filepath to recorded data. If not provided, will use simulated data.')
    parser.add_argument('--use-iterative-solver', type=bool, default=False, help='Use iterative solver (iSAM) instead of batch solver (SAM).')
    parser.add_argument('--num-iters-before-batch', type=int, default=25, help='Number of iterations before running batch solver.')
    parser.add_argument('-p', '--plot-live', type=bool, default=False, action='store_true', help='Plot the factor graph at each timestep.')
    args = vars(parser.parse_args())

    main(**args)
