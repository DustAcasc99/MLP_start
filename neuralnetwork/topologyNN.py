
""" Classes used to define the topology of the Fully-Connected NN
"""

# import the packages
from typing import Callable
import numpy as np
from scipy import misc


# build the Unit class
class Unit:

    """ Class describing a single Unit, either a hidden,
        input or output unit
    """

    def __init__(self, activation_function : Callable[[float], float],
    weights_array : np.ndarray, bias : float, eta : float):

        """ Defining the constructor

        Attributes
        ----------
        activation_function : function
            Activation function to be applied to the net of
            the unit at hand, so it is a one-variable
            function defined outside the network.

        weights_array : arraylike of shape (n_components)
            Weights fot the current unit, taken from the
            weights matrix defined in the corrisponding layer.

        bias : float
            Threshold value of the output unit.

        eta : float
            Learning rate for the alghoritm speed control.

        inputs : arraylike of shape (n_components)
            Set of data (output) from the units of the previous
            layer.

        net : float
            The weighted sum of the inputs to the units at hand.
        """
        # definition of attributes using contructor's arguments
        self.activation_function = activation_function
        self.weights_array = weights_array
        self.bias = bias
        self.eta = eta

        # definition of attributes useful for class methods
        self.counter = 0
        self.gradients_sum = 0.
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

    """ Defining the class for an Output unit
    """

    def __init__(self, activation_function : Callable[[float], float], weights_array: np.ndarray,
        bias: float, eta: float):

        # calls the contructor of the Unit class (the Parent class)
        super().__init__(activation_function, weights_array, bias, eta)


    def backprop_unit(self, target: float, minibatch_size=1) -> float:

        """ Method for the backpropagation of the output unit.

        Arguments
        ----------
        target : float
            The target value relative to the pattern that we've given as input.

        batch_length : int
            Number of samples in batch or mini-batch: it tells after how many calls of
            backprop_unit method the units weights must be updated.
            minibatch _size = 1 ..................... On-line.
            minibatch _size = len(DataSet) .......... Batch.

        Returns
        ----------
        delta : float
            Small delta for the unit, the error signal for the output unit.
        """
        # computes the error signal for the output unit. Note: we use the result
        # of the feedforward process
        delta = ((target - self.output) *
                  misc.derivative(self.activation_function, self.net))

        # sum of gradients and counter update
        self.gradients_sum = self.gradients_sum + delta * self.inputs
        self.counter = self.counter + 1

        # updates the weights (do it only at the end of minibatch)
        if (self.counter == minibatch_size):
            self.weights_array = self.weights_array + (self.eta / minibatch_size) * \
                                    self.gradients_sum
            self.counter = 0
            self.gradients_sum = 0.

        # returns the error signal for this output unit that will be used
        # to compute the backpropagation for the first inner layer of the network
        return delta



class HiddenUnit(Unit):

    """ Defining the class for a Hidden unit
    """

    def __init__(self, activation_function : Callable[[float], float], weights_array: np.ndarray,
        bias: float, eta: float):

        # calls the contructor of the Unit class
        super().__init__(activation_function, weights_array, bias, eta)


    def backprop_unit(self, delta_next : np.ndarray, weights_array_next : np.ndarray,
                        minibatch_size=1) -> float:

        """ Method for the backpropagation of the output unit.

        Arguments
        ----------
        delta_next : arraylike of shape (n_components)
            Array with the deltas computed doing the backpropagation of the units of the
            first outer layer (Note: it's important to remember that we're considering a
            fully-connected NN).

        weights_array_next : arraylike of shape (n_components)
            Array with the weights, connections with the units of the first outer layer.

        batch_length : int
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
        delta = ((np.inner(delta_next, weights_array_next)) *
                  misc.derivative(self.activation_function, self.net))

        # sum of gradients and counter update
        self.gradients_sum = self.gradients_sum + delta * self.inputs
        self.counter = self.counter + 1

        # updates the weights (do it only at the end of minibatch)
        if (self.counter == minibatch_size):
            self.weights_array = self.weights_array + (self.eta / minibatch_size) * \
                                    self.gradients_sum
            self.counter = 0
            self.gradients_sum = 0.

        # returns the error signal for this output unit that will be used
        # to compute the backpropagation for the other hidden units of the network
        return delta



# build the class for a single layer...Note: we differenciate between hidden or output
# layers to take into account the differences in their methods
class OutputLayer:

    """ Class describing the Output layer of the network
    """

    def __init__(self, activation_function : Callable[[float], float], eta: float,
        number_units : int, inputs : np.ndarray):

        """ Defining the constructor

        Attributes
        ----------
        number_units : int
            Number of single units in the output layer.

        inputs : np.ndarray
            Array with the inputs coming from the units of the first inner layer
            to every unit of the output layer.

        eta : float
            Learning rate for the alghoritm speed control.

        weights_matrix : np.ndarray
            Matrix (number of units in the output layer x number of inputs) with the weights,
            connections of every unit of the output layer to those of the first inner layer
            of the network.

        bias_array : np.ndarray
            Array with bias values for every unit of the output layer.

        output_units : list
            List of Output Units that create our output layer.

        layer_outputs : np.ndarray
            Array with the computed outputs of every unit in the output layer.

        layer_delta : np.ndarray
            Array with delta values computed for every unit in the output layer.
        """

        self.number_units = number_units
        self.inputs = inputs
        self.eta = eta

        # initializing the weights_matrix with random values chosen from a uniform
        # distribution and with values from the interval [0,1]
        self.weights_matrix = np.random.uniform(low = 0., high = 1.,
            size = (self.number_units, len(self.inputs)))

        # initializing the values of the bias for every unit of the output layer
        self.bias_array = np.zeros(self.number_units)

        # composition with the single OutputUnit class
        self.output_units = np.array([OutputUnit(activation_function, self.weights_matrix[i, :],
            self.bias_array[i], self.eta) for i in range(self.number_units)])

        # initializing the output values for every unit of the hidden layer
        self.layer_outputs = np.zeros(self.number_units)

        # initializing the array with delta values for every unit of the hidden layer
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


    def backprop_layer(self, target_layer : np.ndarray) -> np.ndarray:

        """ Method for the backpropagation of the output layer.

        Arguments
        ----------
        target_layer : np.ndarray
            Array with the target values for every unit of the output layer.

        Returns
        ----------
        layer_delta : np.ndarray
            Array with delta values computed for every unit in the output layer.
        """
        # computes the delta for every unit in the output layer and updates the weights
        for i in range(self.number_units):
            self.layer_delta[i] =  self.output_units[i].backprop_unit(target_layer[i])
            self.weights_matrix[i, :] = self.output_units[i].weights_array

        return self.layer_delta



class HiddenLayer:

    """ Class describing a single Hidden layer of the network
    """

    def __init__(self, activation_function: Callable[[float], float], eta: float,
        number_units : int, inputs : np.ndarray):

        """ Defining the constructor

        Attributes
        ----------
        number_units : int
            Number of single units in the hidden layer.

        inputs : np.ndarray
            Array with the inputs coming from the units of the first inner layer
            to every unit of the hidden layer at hand.

        eta : float
            Learning rate for the alghoritm speed control.


        weights_matrix : np.ndarray
            Matrix (number of units in the hidden layer x fan in) with the weights,
            connections of every unit of the hidden layer to those of the first inner layer
            of the network.

        bias_array : np.ndarray
            Array with bias values for every unit of the hidden layer.

        hidden_units : list
            List of Hidden Units that create our hidden layer.

        layer_outputs : np.ndarray
            Array with the computed outputs of every unit in the hidden layer.

        layer_delta : np.ndarray
            Array with delta values computed for every unit in the hidden layer.
        """
        self.number_units = number_units
        self.inputs = inputs
        self.eta = eta

        # initializing the weights_matrix with random values chosen from a uniform
        # distribution and with values from the symmetric interval centered in 0 and
        # with width = 1/sqrt(fan_in)
        self.weights_matrix = np.random.uniform(low = - 1/(np.sqrt(len(self.inputs))),
            high = 1/(np.sqrt(len(self.inputs))), size = (self.number_units, len(self.inputs)))

        # initializing the values of the bias for every unit of the hidden layer
        self.bias_array = np.zeros(self.number_units)

        # composition with the single HiddenUnit class
        self.hidden_units = np.array([HiddenUnit(activation_function, self.weights_matrix[i, :],
            self.bias_array[i], self.eta) for i in range(self.number_units)])

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
                        weights_matrix_next : np.ndarray) -> np.ndarray:

        """ Method for the backpropagation of the hidden layer.

        Arguments
        ----------
        delta_next : np.ndarray
            Array with delta values of the first outer layer in the network.

        weights_matrix_next : np.ndarray
            Matrix (number of units in the output layer x number of units in the hidden
            layer) with the weights, connections of every unit of the hidden layer to
            those of the first outer layer of the network.

        Returns
        ----------
        layer_delta : np.ndarray
            Array with delta values computed for every unit in the hidden layer.
        """
        # computes the delta for every unit in the layer at hand and updates the weights
        for i in range(self.number_units):
            self.layer_delta[i] =  self.hidden_units[i].backprop_unit(delta_next, weights_matrix_next[i, :])
            self.weights_matrix[i, :] = self.hidden_units[i].weights_array

        return self.layer_delta
