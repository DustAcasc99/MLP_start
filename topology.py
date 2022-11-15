
""" Python code in which are are defined some classes
for the units, the layers ande the network."""

# import the packages
import numpy as np
from matplotlib import pyplot as plt
import mygrad as mg


# build the unit class
class Unit:
    """ Class describing the unit, either a hidden,
    input or output unit 
    """
    def __init__(self, activation_function : callable[float, float],
    weights_array : np.ndarray, bias : float, eta : float):
        """ Defining the constructor.

        Attributes
        ----------
        activation_function : callable[float, float]
            Activation function to be applied to net of
            the current unit, so it is a one variable
            function defined outside the network.

        weights_array : arraylike of shape (n_components)
            Weights fot the current unit, taken from the
            weights matrix defined in the corrisponding layer.

        bias : float
            Treshold of the output unit.

        eta : float
            Learning rate for the alghoritm speed control.
        """
        self.activation_function = activation_function
        self.weights_array = weights_array
        self.bias = bias
        self.eta = eta

    def feedforward_unit(self, inputs : np.ndarray) -> float:
        """ Method for the forward propagation of the unit.

        Arguments
        ----------
        inputs_ : arraylike of shape (n_components)
            Set of data in input (output from the units of the 
            previous layer). The lenght of this vector is the fan in.

        Return
        ----------
        self.unit_output : float
            Coputation of the unit output.
        """
        # load the input int hte attribute
        self.inputs = inputs

        # computing the net and the output of the unit
        self.net = np.inner(self.weights_array, self.inputs) + self.bias
        self.unit_output = self.activation_function(self.net) 

        return self.unit_output



class OutputUnit(Unit):
    """ Defining the class of the output unit.
    """

    def __init__(self):

        # call the contructor of the father class
        Unit.__init__()


    def backprop_unit(self, target: float) -> float:
        """ Method for the backpropagation of the output unit.

        Arguments
        ----------
        target : float
            Label of the present sample for the output unit.

        Return
        ----------
        self.delta : float
            Small delta for the unit (delta_k).
        """
        # compute the error signal for the output unit
        activationf = self.activation_function(mg.tensor(self.net))
        delta = (target - self.unit_output) * activationf.backward()

        ### I ried to run the commands for the drivative on shell, but 
        ### something like func.backward() is a NoneType and not a number.
        ### Also the term backward is something not well to read in a
        ### method for the backward propagation. (Andrea)

        # update the weights
        self.weights = self.weights + self.eta *  delta * self.inputs

        return delta



class HiddenUnit(Unit):
    """ Defining the class of a hidden Unit.
    """

    def __init__(self):

        # call the contructor of the Father class
        Unit.__init__()

    def backprop_unit(self, delta_next, weights_array_next) -> float:
        """ Method for the backpropagation of the output unit.

        Arguments
        ----------
        delta_next : arraylike of shape (n_components)
            Deltas calculated whith backpropagation for the
            next layer in the network.

        weights_array: arraylike of shape (n_components)
            Weights of the synapses linked to the output of
            the present unit.

        Return
        ----------
        self.delta : float
            Small delta for the unit (delta_j).
        """
        # compute the error signal for the hidden unit
        activationf = self.activation_function(mg.tensor(self.net))
        delta = (np.inner(delta_next, weights_array_next)) * activationf.backward()

        # update the weights
        self.weights_array = self.weights_array + self.eta * delta * self.inputs

        return delta


class Layer:
    """ Class describing a single layer of the network
    """
    def __init__(self, number_units : int):
        self.number_units = number_units
        self.hidden_unit = HiddenUnit()
        self.output_unit = OutputUnit()
