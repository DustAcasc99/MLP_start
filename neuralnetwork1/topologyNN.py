
""" Classes used to define the topology of a Fully-Connected NN.
"""

# import the packages
from typing import Callable
import numpy as np
from scipy import misc


# build the Unit class
class Unit:

    """ Class describing a single unit, 
        either a hidden or output unit.
    """

    def __init__(self, activation_function : Callable[[float], float],
                weights_array : np.ndarray, bias : float, seed : int, eta_0 : float, 
                alpha = 0., lamb = 0., lamb0 = 0.):

        """ Defining the constructor.

        Attributes
        ----------
        activation_function : function
            Activation function to be applied to the net of
            the unit, so it is a one-variable function
            defined outside the network.

        weights_array : arraylike of shape (n_components)
            Weights fot the current unit, to be taken from the
            weights matrix defined in the corrisponding layer.

        bias : float
            Threshold value of the output unit.

        seed : int
            seed for the random generation of the weights.

        eta_0 : float
            Maximum learning rate for the alghoritm speed control.

        alpha : float
            Coefficient for momentum implementation, with value None
            if not implemented or a generic numbers pass from user.

        lamb : float
            Lambda in the penalty term for regularization (word lambda
            is not used because it is a reserved word in Python).

        lamb0 : float
            The same as for lamb but for the bias term.
        """
        # definition of attributes using contructor's arguments
        self.activation_function = activation_function
        self.weights_array = weights_array
        self.bias = bias
        self.seed = seed
        self.eta_0 = eta_0
        self.alpha = alpha
        self.lamb = lamb
        self.lamb0 = lamb0

        # definition of attributes useful for class methods
        self.eta = float
        self.tau = 250
        self.counter = 0
        self.gradients_sum = 0.
        self.old_weight_change = 0.
        self.gradients_sum_bias = 0.
        self.old_bias_change = 0.
        self.inputs = np.ndarray
        self.output = np.ndarray
        self.net = float


    def feedforward_unit(self, inputs : np.ndarray) -> float:

        """ Method for the forward propagation of the unit.

        Arguments
        ----------
        inputs : arraylike of shape (n_components)
            Set of data (output) from the units of the previous
            layer.

        Returns
        ----------
        self.unit_output : float
            The computed unit output.
        """
        # saving the inputs (also for backpropagation)
        self.inputs = inputs

        # computing the net and the output of the unit
        self.net = np.inner(self.weights_array, self.inputs) + self.bias
        self.output = self.activation_function(self.net)

        return self.output



# Note: build two different classes for the output and hidden units that inherit
# from the Unit class because the computation for the backpropagation is different
class OutputUnit(Unit):

    """ Defining the class for an output unit.
    """

    def __init__(self, activation_function : Callable[[float], float], weights_array : np.ndarray,
        bias : float, seed : int, eta_0 : float, alpha = 0., lamb = 0., lamb0 = 0.):

        # calls the contructor of the Unit class (the Parent class)
        super().__init__(activation_function, weights_array, bias, seed, eta_0, alpha, lamb, lamb0)


    def backprop_unit(self, target: float, minibatch_size = 1) -> float:

        """ Method for the backpropagation of the output unit.

        Arguments
        ----------
        target : float
            The target value relative to the pattern that we've given as input.

        minibatch_size : int
            Number of samples in batch or mini-batch: it tells after how many calls of
            the backprop_unit method the units weights must be updated.
            minibatch _size = 1 ..................... On-line.
            minibatch _size = len(DataSet) .......... Batch.

        Returns
        ----------
        delta : float
            Small delta for the unit, the error signal for the output unit.
        """
        # computing the error signal for the output unit. Note: we use the result
        # of the feedforward process
        delta = ((target - self.output) *
                  misc.derivative(self.activation_function, self.net))

        # summing of gradients and counter update
        self.gradients_sum = self.gradients_sum + delta * self.inputs
        self.gradients_sum_bias = self.gradients_sum_bias + delta * self.bias
        self.counter += 1

        # implementing the variable learning rate at the first order
        if self.counter < self.tau:
            self.eta = (1 - self.counter / self.tau) * self.eta_0 + \
                       (self.counter / self.tau) * 0.01 * self.eta_0
        else:
            self.eta = 0.01 * self.eta_0

        # updating the weights (do it only at the end of minibatch)
        if (self.counter == minibatch_size):
            self.weights_array = self.weights_array + \
                                    (self.eta / minibatch_size) * self.gradients_sum + \
                                    self.alpha * self.old_weight_change - \
                                    self.lamb * self.weights_array
            self.bias = self.bias + (self.eta / minibatch_size) * self.gradients_sum_bias + \
                                    self.alpha * self.old_bias_change - \
                                    self.lamb0 * self.bias
            
            # updating the momentum
            self.old_weight_change = (self.eta / minibatch_size) * self.gradients_sum
            self.old_bias_change = (self.eta / minibatch_size) * self.gradients_sum_bias

            # resetting quantities for next minibatch
            self.counter = 0
            self.gradients_sum = 0.
            self.gradients_sum_bias = 0.

        # returns the error signal for this output unit that will be used
        # to compute the backpropagation for the first inner layer of the network
        return delta



class HiddenUnit(Unit):

    """ Defining the class for a hidden unit.
    """

    def __init__(self, activation_function : Callable[[float], float], weights_array : np.ndarray,
        bias : float, seed : int, eta_0 : float, alpha = 0., lamb = 0., lamb0 = 0.):

        # calls the contructor of the Unit class
        super().__init__(activation_function, weights_array, bias, seed, eta_0, alpha, lamb, lamb0)
 

    def backprop_unit(self, delta_next : np.ndarray, weights_array_next : np.ndarray,
                      bias_next : float, minibatch_size = 1) -> float:

        """ Method for the backpropagation of the output unit.

        Arguments
        ----------
        delta_next : arraylike of shape (n_components)
            Array with the deltas computed doing the backpropagation of the units of the
            first outer layer (Note: it's important to remember that we're considering a
            fully-connected NN).

        weights_array_next : arraylike of shape (n_components)
            Array with the weights, connections with the units of the first outer layer.

        bias_next : float
            Value of the bias with the units of the first outer layer.

        minibatch_size : int
            Number of samples in batch or mini-batch: it tells after how many calls of
            backprop_unit method the units weights must be updated.
            minibatch _size = 1 ..................... On-line.
            minibatch _size = len(DataSet) .......... Batch.

        Returns
        ----------
        delta : float
            Small delta for the unit, the error signal for the hidden unit.
        """
        # computes the error signal for the hidden unit
        delta_weights = ((np.inner(delta_next, weights_array_next)) *
                  misc.derivative(self.activation_function, self.net))
        delta_bias = ((np.inner(delta_next, bias_next)) *
                  misc.derivative(self.activation_function, self.net))
        delta = delta_weights + delta_bias

        # sum of gradients and counter update
        self.gradients_sum = self.gradients_sum + delta_weights * self.inputs
        self.gradients_sum_bias = self.gradients_sum_bias + delta_bias * self.bias
        self.counter += 1

        # implementing the variable learning rate at the first order
        if self.counter < self.tau:
            self.eta = (1 - self.counter / self.tau) * self.eta_0 + \
                       (self.counter / self.tau) * 0.01 * self.eta_0
        else:
            self.eta = 0.01 * self.eta_0

        # updates the weights (do it only at the end of minibatch)
        if (self.counter == minibatch_size):
            self.weights_array = self.weights_array + \
                                    (self.eta / minibatch_size) * self.gradients_sum + \
                                    self.alpha * self.old_weight_change - \
                                    self.lamb * self.weights_array
            self.bias = self.bias + (self.eta / minibatch_size) * self.gradients_sum_bias + \
                                    self.alpha * self.old_bias_change - \
                                    self.lamb0 * self.bias

            # update the momentum
            self.old_weight_change = (self.eta / minibatch_size) * self.gradients_sum
            self.old_bias_change = (self.eta / minibatch_size) * self.gradients_sum_bias

            # reset quantities for next minibatch (or sample/epoch)
            self.counter = 0
            self.gradients_sum = 0.
            self.gradients_sum_bias = 0.

        # returns the error signal for this output unit that will be used
        # to compute the backpropagation for the other hidden units of the network
        return delta



# build the class for a single layer...Note: we differenciate between hidden or output
# layers to take into account the differences in their methods
class OutputLayer:

    """ Class describing an output layer of the network.
    """

    def __init__(self, activation_function : Callable[[float], float],
                number_units : int, inputs : np.ndarray, seed : int,
                eta_0 : float, alpha = 0., lamb = 0., lamb0 = 0.):

        """ Defining the constructor

        Attributes
        ----------
        activation_function : function
            Activation function to be applied to the net of
            the unit, so it is a one-variable function
            defined outside the network.

        number_units : int
            Number of single units in the output layer.

        inputs : np.ndarray
            Array with the inputs coming from the units of the first inner layer
            to every unit of the output layer.

        seed : int
            seed for the random generation of the weights.

        eta_0 : float
            Maximum learning rate for the alghoritm speed control.

        alpha : NoneType or float
            Coefficient for momentum implementation, with value None
            if not implemented or a generic numbers pass from user.

        lamb : float
            Lambda in the penalty term for regularization (word lambda
            is not used because it is a reserved word in Python).

        lamb0 : float
            The same as for lamb but for the bias term.
        """
        # definition of attributes using contructor's arguments
        self.number_units = number_units
        self.inputs = inputs
        self.seed = seed
        self.eta_0 = eta_0
        self.alpha = alpha
        self.lamb = lamb
        self.lamb0 = lamb0

        # initializing the weights_matrix with random values chosen from a uniform
        # distribution and with values from the interval [-0.7,0.7]
        self.weights_matrix = np.random.RandomState(seed).uniform(low = -0.7, high = 0.7,
            size = (self.number_units, len(self.inputs)))

        # initializing the values of the bias for every unit of the output layer
        self.bias_array = np.zeros(self.number_units)

        # composition with the single OutputUnit class
        self.output_units = np.array([OutputUnit(activation_function, self.weights_matrix[i, :], self.bias_array[i],
                            self.seed, self.eta_0, self.alpha, self.lamb, self.lamb0) for i in range(self.number_units)])

        # initializing the output values for every unit of the output layer
        self.layer_outputs = np.zeros(self.number_units)

        # initializing the array with delta values for every unit of the output layer
        self.layer_delta = np.zeros(self.number_units)


    def feedforward_layer(self) -> np.ndarray:

        """ Method for the forward propagation of the output layer.

        Returns
        ----------
        layer_outputs : np.ndarray
            Array with the computed outputs of every unit in the output layer.
        """
        # computes the output of all the units in the output layer and collects
        # them in a numpy array (Note: we're considering a fully-connected NN, else we couldn't
        # take the same inputs for all the units in the layer at hand)
        for i in range(self.number_units):
            self.layer_outputs[i] =  self.output_units[i].feedforward_unit(self.inputs)

        return self.layer_outputs


    def backprop_layer(self, target_layer : np.ndarray, minibatch_size = 1) -> np.ndarray:

        """ Method for the backpropagation of the output layer.

        Arguments
        ----------
        target_layer : np.ndarray
            Array with the target values for every unit of the output layer.

        minibatch_size : int
            Number of samples in batch or mini-batch: it tells after how many calls of
            backprop_unit method the units weights must be updated.
            minibatch _size = 1 ..................... On-line.
            minibatch _size = len(DataSet) .......... Batch.

        Returns
        ----------
        layer_delta : np.ndarray
            Array with delta values computed for every unit in the output layer.
        """
        # computes the delta for every unit in the output layer and updates the weights
        for i in range(self.number_units):
            self.layer_delta[i] =  self.output_units[i].backprop_unit(target_layer[i], minibatch_size)
            self.weights_matrix[i, :] = self.output_units[i].weights_array
            self.bias_array[i] = self.output_units[i].bias

        return self.layer_delta



class HiddenLayer:

    """ Class describing a single Hidden layer of the network
    """

    def __init__(self, activation_function : Callable[[float], float],
                number_units : int, inputs : np.ndarray, seed : int,
                eta_0 : float, alpha = 0., lamb = 0., lamb0 = 0.):

        """ Defining the constructor

        Attributes
        ----------
        activation_function : function
            Activation function to be applied to the net of
            the unit, so it is a one-variable function
            defined outside the network.
        
        number_units : int
            Number of single units in the hidden layer.

        inputs : np.ndarray
            Array with the inputs coming from the units of the first inner layer
            to every unit of the hidden layer at hand.

        seed : int
            seed for the random generation of the weights.

        eta_0 : float
            Maximum learning rate for the alghoritm speed control.

        alpha : float
            Coefficient for momentum implementation, with value None
            if not implemented or a generic numbers pass from user.

        lamb : float
            Lambda in the penalty term for regularization (word lambda
            is not used because it is a reserved word in Python).

        lamb0 : float
            The same as for lamb but for the bias term.
        """

        # definition of attributes using contructor's arguments
        self.number_units = number_units
        self.inputs = inputs
        self.seed = seed
        self.eta_0 = eta_0
        self.alpha = alpha
        self.lamb = lamb
        self.lamb0 = lamb0

        # initializing the weights_matrix with random values chosen from a uniform
        # distribution and with values from the symmetric interval centered in 0 and
        # with width = 1/sqrt(fan_in)
        self.weights_matrix = np.random.RandomState(seed).uniform(low = - 1/(np.sqrt(len(self.inputs))),
            high = 1/(np.sqrt(len(self.inputs))), size = (self.number_units, len(self.inputs)))

        # initializing the values of the bias for every unit of the hidden layer
        self.bias_array = np.zeros(self.number_units)

        # composition with the single HiddenUnit class
        self.hidden_units = np.array([HiddenUnit(activation_function, self.weights_matrix[i, :],
            self.bias_array[i], self.seed, self.eta_0, self.alpha, self.lamb, self.lamb0) for i in range(self.number_units)])

        # initializing the output values for every unit of the hidden layer
        self.layer_outputs = np.zeros(self.number_units)

        # initializing the array with delta values for every unit of the hidden layer
        self.layer_delta = np.zeros(self.number_units)


    def feedforward_layer(self) -> np.ndarray:

        """ Method for the forward propagation of the hidden layer.

        Returns
        ----------
        layer_outputs : np.ndarray
            Array with the computed outputs of every unit in the hidden layer.
        """
        # computes the output of all the units in the layer at hand and collects
        # them in a numpy array (Note: we're considering a fully-connected NN, else we couldn't
        # take the same inputs for all the units in the layer at hand)
        for i in range(self.number_units):
            self.layer_outputs[i] =  self.hidden_units[i].feedforward_unit(self.inputs)

        return self.layer_outputs


    def backprop_layer(self, delta_next : np.ndarray,
                        weights_matrix_next : np.ndarray, bias_array_next : np.ndarray,
                        minibatch_size = 1) -> np.ndarray:

        """ Method for the backpropagation of the hidden layer.

        Arguments
        ----------
        delta_next : np.ndarray
            Array with delta values of the first outer layer in the network.

        weights_matrix_next : np.ndarray
            Matrix (number of units in the output layer x number of units in the hidden
            layer) with the weights, connections of every unit of the hidden layer to
            those of the first outer layer of the network.

        bias_array_next : np.ndarray
            Array with bias values with thw units of the first outer layer.

        minibatch_size : int
            Number of samples in batch or mini-batch: it tells after how many calls of
            backprop_unit method the units weights must be updated.
            minibatch _size = 1 ..................... On-line.
            minibatch _size = len(DataSet) .......... Batch.

        Returns
        ----------
        layer_delta : np.ndarray
            Array with delta values computed for every unit in the hidden layer.
        """
        # computes the delta for every unit in the layer at hand and updates the weights
        for i in range(self.number_units):
            self.layer_delta[i] =  self.hidden_units[i].backprop_unit(delta_next, 
                                    weights_matrix_next[:, i], bias_array_next, minibatch_size)
            self.weights_matrix[i, :] = self.hidden_units[i].weights_array
            self.bias_array[i] = self.hidden_units[i].bias

        return self.layer_delta
