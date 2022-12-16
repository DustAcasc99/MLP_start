
""" File to perform the training, validation and test phases.
"""

# import packages 
import numpy as np

# import files
from train_val import performing_tv, linear, sigmoid


# loading training and validation data in a numpy array skipping the first 7 rows
tvs_array_load = np.loadtxt('ML-CUP22-TR.csv', skiprows = 6, delimiter = ',',
                             usecols = (1,2,3,4,5,6,7,8,9,10,11))

# normalizing inputs
tvs_array_load_mv = np.zeros((len(tvs_array_load[:, 0]), 9))
mean_array = np.zeros(9)
standard_deviation_array = np.zeros(9)
for i in range(9):
    mean_array[i] = np.average(tvs_array_load[:, i])
    standard_deviation_array[i] = np.std(tvs_array_load[:, i])

for i in range(len(tvs_array_load[:, 0])):
    for j in range(9):
        tvs_array_load_mv[i, j] = (tvs_array_load[i, j] - mean_array[j]) / standard_deviation_array[j]

tvs_array_load_mv = np.column_stack((tvs_array_load_mv, tvs_array_load[:, -2], tvs_array_load[:, -1]))

tvs_array_load_minmax = np.zeros((len(tvs_array_load[:, 0]), 9))
min_array = np.zeros(9)
max_array = np.zeros(9)
for i in range(9):
    min_array[i] = np.amin(tvs_array_load[:, i])
    max_array[i] = np.amax(tvs_array_load[:, i])

for i in range(len(tvs_array_load[:, 0])):
    for j in range(9):
        tvs_array_load_minmax[i, j] = (tvs_array_load[i, j] - min_array[j]) / (max_array[j] - min_array[j])

tvs_array_load_minmax = np.column_stack((tvs_array_load_minmax, tvs_array_load[:, -2], tvs_array_load[:, -1]))
# --------------------

layers_list = performing_tv([2, 3], [8, 10, 12], 9, 2, tvs_array_load_mv, [3, 4, 5], [0.3], [0.5, 0.7, 0.9],
                            [0.001, 0.01], [0.001, 0.01], 4,
                            [30, 50, 70], linear, sigmoid,  stop_class = 'GL', stop_param = 3,
                            task = 'regression', thr = 0)