# EC EN 633 Class Project - Python iSAM Implementation

This repository is a Python implementation of the iSAM (incremental smoothing and mapping) algorithm for SLAM.
The intent of this project is to learn and demonstrate an understanding of the iSAM algorithm.
As such, we do not use any existing factor graph libraries and are not trying to run this in real-time.

## Style Guide

Follow the PEP 8 styleguide in addition to these rules.
1. Private variables, methods, classes, etc. of a module should have a leading underscore. (i.e. `_private_function`, `_PrivateClass`)
2. All non-private methods in modules need to have typing enforced for both the function arguments and return values. This will likely need to use `assert` statements, due to numpy's lack of typing enforcement.
3. All project dependencies need to be listed in `requirements.txt`.

Also, any vector that is typically represented as a column vector in mathematics should be stored as a 2D Nx1 vector in numpy, not a 1D vector.
This is including measurements, states, and odometry. When multiple of these are stacked together, form a NxM matrix where N is the length of the individual vector and M is the number of vectors you've stacked together.
This ensures consistency of our code with itself and the mathematical theory, even if using 2D arrays is more verbose.
