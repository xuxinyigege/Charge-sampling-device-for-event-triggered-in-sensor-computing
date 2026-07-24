# Data

- `patterns.csv`: ideal 4 x 4 binary Z/J/U patterns.
- `zju_10db_dataset.npz`: noisy 10 dB dataset converted from the legacy MATLAB
  `.mat` files.

The NumPy archive stores:

- `X`: array with shape `(1500, 4, 4)`;
- `y`: integer labels where `Z = 0`, `J = 1`, `U = 2`;
- `class_names`: class-name array.
