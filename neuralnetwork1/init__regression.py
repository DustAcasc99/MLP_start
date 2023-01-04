
""" File to perform the training, validation and test phases.
"""

# import packages 
import numpy as np

# import files
from train_val import performing_tvt, linear, sigmoid, ReLU, ELU, swish, train_test


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

# normalizing inputs and outputs
tvs_array_load_mv = np.zeros((len(tvs_array_load[:, 0]), 11))
mean_array = np.zeros(11)
standard_deviation_array = np.zeros(11)
for i in range(11):
    mean_array[i] = np.average(tvs_array_load[:, i])
    standard_deviation_array[i] = np.std(tvs_array_load[:, i])

for i in range(len(tvs_array_load[:, 0])):
    for j in range(11):
        tvs_array_load_mv[i, j] = (tvs_array_load[i, j] - mean_array[j]) / standard_deviation_array[j]


tvs_array_load_minmax = np.zeros((len(tvs_array_load[:, 0]), 11))
min_array = np.zeros(11)
max_array = np.zeros(11)
for i in range(11):
    min_array[i] = np.amin(tvs_array_load[:, i])
    max_array[i] = np.amax(tvs_array_load[:, i])

for i in range(len(tvs_array_load[:, 0])):
    for j in range(11):
        tvs_array_load_minmax[i, j] = (tvs_array_load[i, j] - min_array[j]) / (max_array[j] - min_array[j])

'''
results = performing_tvt(layers_range=[1], units_range=[25], num_inputs=9,
                            num_targets=2, tvts_array=tvs_array_load_minmax, k_range=[3], eta_0_range=[0.15],
                            alpha_range=[0.8], lamb_range=[0.0001], lamb0_range=[0.0001],
                            configurations=4, minibatch_size_range=[100], activation_output=linear,
                            activation_hidden=swish, stop_class='GL', stop_param=2,
                            task='regression', thr=0)

for index in range(len(results[1])):
    print(f'Set up: {results[1][index]}, val error: {results[2][index]}, test error: {results[3][index]}')
'''

hyperparams = {
    'layers' : 5,
    'units' : [10, 10, 10, 10, 2],
    'eta_0' : 0.15,
    'alpha' : 0.8,
    'lamb' : 0,
    'lamb0' : 0,
    'minibatch_size' : 100
}

results = train_test(hyperparams=hyperparams, num_inputs=9, seed=1, activation_output=linear, 
                     activation_hidden=swish, task='regression', thr=0.5, stop_class='PQ', stop_param=8,
                     data_train=tvs_array_load_minmax[:int(0.9*len(tvs_array_load_minmax))],
                     data_val=tvs_array_load_minmax[int(0.9*len(tvs_array_load_minmax)):], 
                     scale_factor = (max_array[j] - min_array[j]), shift = min_array[j] )

print(f'Set up: {hyperparams}, test error: {results[2]}, train error: {results[3]}')
