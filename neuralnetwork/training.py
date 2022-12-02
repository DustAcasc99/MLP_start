import numpy as np
from topologyNN import OutputUnit, HiddenUnit, OutputLayer, HiddenLayer

import math

    
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def linear(x):
    return x
    



def reset_mini_batch(layers):
    for layer in layers[:-1]:
        for hidden_unit in layer.hidden_units:
            hidden_unit.counter = 0
            hidden_unit.gradients_sum = 0
            
    for output_unit in layers[-1].output_units:
        output_unit.counter = 0
        output_unit.gradients_sum = 0
        
        

def feedforward_train(layers, to_pass):
    """ Parameters for the feedforward_train auxiliary function:
        
        layers : list
            list of layers defined in the main function
            
        to_pass : np.array
            inputs row from the data-set to be passed as parameter
            to the first layer in "layers"

    Returns
    ----------
    output : np.ndarray
        Array with the output of the final output_layer
    """
    
    #perform a forward propagation for all the layers in "layers"
    for layer_index in range(len(layers)):
        #consider the layer of index "layer_index" and update its
        #internal inputs with "to_pass"        
        layers[layer_index].inputs = to_pass
        
        #perform the feedforward on the current layer save its
        #output in order to update the inputs of the next layer
        to_pass = layers[layer_index].feedforward_layer()
    
    
    return to_pass #corresponding to the output of output layer
        
        
def backprop_train(layers, target_layer, minibatch_size):
    """ Parameters for the feedforward_train auxiliary function:
        
        layers : list
            list of layers defined in the main function
            
        target_layer : np.array
            target array of the current input row considered in the
            previous feedforward step.
            
        minibatch_size : int
            the size of the mini-batch considered for the weights
            update inside each layers' units

    Returns
    ----------
    error : float
        the error computed as 1/2(target-output)**2
        
    """
    
    #perform a backpropagation for all the layers in "layers"    
    
    #obtain the delta from the output layer 
   
    error = np.average((target_layer - layers[-1].layer_outputs) ** 2)

    delta = layers[-1].backprop_layer(target_layer = target_layer, minibatch_size=minibatch_size)
    print(target_layer, layers[-1].layer_outputs, error)
    #loop over the remaining hidden layers in reverse
    for layer_index in range(len(layers)-2,-1,-1): #h1 h2 h3 o 
        
        #take matrix of upper layer        
        weights_next = layers[layer_index+1].weights_matrix 
        #call backprop the layer and update the delta for the further layers
        delta = layers[layer_index].backprop_layer(delta_next = delta, weights_matrix_next = weights_next, minibatch_size = minibatch_size)
    
    return error

def train(X_train, y_train, eta, num_layers = 1,  units_per_layer = [2], minibatch_size = 1, epochs = 1, alpha = 0.0, lamb = 0.0,
          activation_hidden = sigmoid, activation_output = linear):
    """ Parameters for the feedforward_train auxiliary function:
        
        X_train : list
            list of input pattern, each pattern being a np.array
        
        y_train : list
            list of target of the respective pattern,
            each target being a np.array
            
        num_layers : int
            number of hidden_layers and output_layer to train
            (if 1, a single output_layer is trained)
            
        units_per_layer : list
            list of integers specifying for each layer how many units
            to instantiate
        
        minibatch_size : int
            specifies the size of the minibatch according to which 
            the weights update must be performed
            
        epochs : int
            specifies the number of total training epochs
            
    Returns
    ----------
    layers : list
        list of trained hidden layers and output layer 
        (weights matrices already trained)
            

    """
    
   
    
    
    if len(units_per_layer) != num_layers:
        raise ValueError('Lenght of units_per_layer should be equal to num_layers')
    
    
    to_pass = X_train[0]
    
    layers = []
    
 
    for i in range(0,num_layers-1):
        
        
        h = HiddenLayer(activation_function = activation_hidden, eta = eta,
                                   number_units = units_per_layer[i], inputs = to_pass, alpha = alpha , lamb = lamb)
        
        to_pass = h.feedforward_layer()
        
        layers += [h]
  
        
        
    

    
    output_layer = OutputLayer(activation_function = activation_output, eta = eta,
                               number_units = units_per_layer[-1], inputs = to_pass, alpha = alpha, lamb = lamb)
    
    output_layer.feedforward_layer()
      

    layers += [output_layer]
    
    #suppose you have 2 hidden and 1 output
    #the first hidden takes in original inputs, run the feedforward, gets appendend and
    #now to_pass = feedforward first layers; then the secodn hidden takes the to_pass
    #run feedforward, gets appendend and before it updates the to pass

    #the last to pass is passed at the output layer, on which we run the feedforward
    #for the further inputs, we need to proceed with the feedforward
    
    errs = []
    for epoch in range(epochs):
        #reset mini_batch at each epochs
        reset_mini_batch(layers)
        epoch_error = 0
        
        for idx in range(0,len(X_train)):
            #print(idx)
            to_pass = X_train[idx]
            output = feedforward_train(layers, to_pass)
            
            pattern_error = backprop_train(layers, y_train[idx], minibatch_size = minibatch_size)
            
            epoch_error += pattern_error
        print('***************')
        
        
        
        #
    return layers
            
    
    
    

if __name__ == "__main__":
        
    import pandas as pd
    
    
    tvs = pd.read_csv('ML-CUP22-TR.csv', skiprows=6, header=0,
          names= ['exemples', 'input_1', 'input_2', 'input_3', 'input_4', 'input_5', 'input_6', 'input_7', 'input_8', 
          'input_9', 'target_1', 'target_2'], 
          index_col= ['exemples'])
    
    # loading it in a numpy array skipping the first 7 rows
    tvs_array = np.loadtxt('ML-CUP22-TR.csv', skiprows=6, delimiter=',')
    
    X_train = [array[1:9] for array in tvs_array]
    
    y_train = [array[10:] for array in tvs_array] 
    
    
    train(X_train[0:10], y_train[0:10], eta = 0.1, num_layers = 2,  units_per_layer = [2,2], minibatch_size = 4, epochs = 100) 
    
    
    
    
    
    
'''



evaluate(layers, X, y):
    square_sum = 0
    for idx in range(0,len(X)):
        nn_output = feedforward_train(layers, X[idx] )
        pattern_error = y[idx] - nn_output
        square_sum += pattern_error ** 2
    
    return square_sum / len(X)


predict_accuracy(layers, X , y):
    match = 0
    for idx in range(0,len(X)):
        nn_output = feedforward_train(layers, X[idx])
        nn_output == y[idx]
        match += 1
    return match / len(X)
        
                
        
for prediction in predict(layers, X, y)
    





'''
        

    


