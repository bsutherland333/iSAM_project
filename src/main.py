#!/usr/bin/env python3

import argparse
import numpy as np

from data_parser import DataParser
from factor_graph_manager import FactorGraphManager
from models import odometry_model, inverse_odometry_model, sensor_model
from simulator import Simulator

# Change what is imported to change the methods
from column_reorderers import colamd as reorder
from iterative_matrix_augmenters import augment_r as augment_r
from qr_factorizers import np_qr as qr
from solvers import np_solver as solver


def main(num_iterations: int, data_filepath: str, use_iterative_solver: bool, num_iters_before_batch: int):
    if data_filepath != '':
        data = DataParser(data_filepath)
        x = data.get_initial_state()
    else:
        x = np.array([[0, 0, 0]]).T
        data = Simulator(inverse_odometry_model, x)

    factor_graph_manager = FactorGraphManager(odometry_model, sensor_model, x)

    for timestep in range(num_iterations):
        # Run the simulator/data parser
        u, z = data.get_next_timestep()

        # Add the measurements to the factor graph and get the A matrix
        for i in range(z.shape[1]):
            factor_graph_manager.add_measurement(z[:, i].reshape(-1, 1))
        factor_graph_manager.add_odometry(u)

        if not use_iterative_solver or timestep % num_iters_before_batch == 0:
            # Solve the linear system with a batch update
            A, b = factor_graph_manager.get_A_b_matrix(x)
            A_prime, P = reorder(A)
            Q, R = qr(A_prime)
            x_prime = solver(Q, R, b)
            x = P @ x_prime
        else:
            # Solve the linear system iteratively
            # TODO: I don't really know how this works yet, so this is probably wrong
            for i in range(z.shape[1]):
                R = augment_r(R, z[:, i].reshape(-1, 1))
            x_prime = solver(Q, R, b)
            x = P @ x_prime

        print(f"timestep: {timestep}, x: {x.flatten()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-iterations', type=int, default=200, help='Number of iterations to run the algorithm for.')
    parser.add_argument('-f', '--data-filepath', type=str, default='', help='Filepath to recorded data. If not provided, will use simulated data.')
    parser.add_argument('--use-iterative-solver', type=bool, default=False, help='Use iterative solver (iSAM) instead of batch solver (SAM).')
    parser.add_argument('--num-iters-before-batch', type=int, default=25, help='Number of iterations before running batch solver.')
    args = vars(parser.parse_args())

    main(**args)
