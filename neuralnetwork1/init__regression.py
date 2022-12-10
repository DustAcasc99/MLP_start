
""" File to perform the training, validation and test phases.
"""

# import packages 
import math
import numpy as np

# import files
from train_val import performing_tv, feedforward_network, network_initialization, backprop_network, train, split_tvs_kfCV
from topologyNN import HiddenLayer


# loading training and validation data in a numpy array skipping the first 7 rows
tvs_array_load = np.loadtxt('ML-CUP22-TR.csv', skiprows = 6, delimiter = ',',
                             usecols = (1,2,3,4,5,6,7,8,9,10,11))

# defining activation functions
def linear(x):
    return x


# --------------------
"""
lista_layers = network_initialization(3, [5,3,2], 9, 0.07, 0.6, 0.0001, linear)

output = feedforward_network(lista_layers, tvs_array_load[0,:9])
print(output)
print(lista_layers[1].weights_matrix)

pattern_error = backprop_network(lista_layers, tvs_array_load[0,9:], 1)
print(pattern_error)
print(lista_layers[1].weights_matrix)

lista_layers, epoch_error = train(tvs_array_load, lista_layers, 9, 30)
print(epoch_error)

a = np.array(([1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4], [5,5,5,5]))
folds_data = split_tvs_kfCV(a, 3)
for i in range(len(folds_data)):
    print(folds_data[i])
"""

lista = performing_tv(layers_range=[2, 3], units_range=[3, 5, 8], eta_range=[0.01, 0.05],
                        alpha_range=[0.4, 0.6], lambda_range=[0.001, 0.01],
                        num_targets=2, tvs_array=tvs_array_load, k=4, minibatch_size=50, 
                        num_inputs=9, activation_output=linear)
