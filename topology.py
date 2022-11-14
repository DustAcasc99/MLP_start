# import the packages
import numpy as np
from matplotlib import pyplot as plt
import mygrad as mg


# build the Unit class
class Unit:
    """ Class describing the unit, either a hidden,
    input or output unit 
    """
    def __init__(self, activation_function : function,
    weights_array : np.ndarray, eta : float):
        """ Defining the constructor.
        """
        self.activation_function = activation_function
        self. weights_array = weights_array
        self.eta = eta

    def feedforward_unit(self, inputs_ : np.ndarray):
        """ Computing the output value
        """
        self.inputs_ = inputs_

        # computing the net and the output of the unit
        net = np.inner(self.weights_array, self.inputs_)
        unit_output = self.activation_function(net)

        return [float(net), float(unit_output)]



class OutputUnit(Unit):
    """ Defining the class of the Output Unit
    """

    def __init__(self, target_ : float, output_prevlayer : np.ndarray):
        # call the contructor of the Father class
        Unit.__init__()

        self.target_ = target_
        self.output_prevlayer = output_prevlayer


    def backprop_unit(self):
        
        # compute the error signal for the output unit
        net = mg.tensor(self.feedforward_unit[0])
        activationf = self.activation_function(net)
        delta = (self.target_ - self.feedforward_unit[1])*activationf.backward()

        # update the weights
        for i in enumerate(self.weights_array):
            self.weights_array[i] = (self.weights_array[i] 
            + self.eta * delta * self.output_prevlayer[i])

        return float(delta)



class HiddenUnit(Unit):
    """ Defining the class of a Hidden Unit
    """

    def __init__(self, delta_output : np.ndarray, output_prevlayer : np.ndarray):
        # call the contructor of the Father class
        Unit.__init__()

        self.delta_output = delta_output
        self.output_prevlayer = output_prevlayer

    def backprop_unit(self):
        
        # compute the error signal for the hidden unit
        net = mg.tensor(self.feedforward_unit[0])
        activationf = self.activation_function(net)
        delta = (np.inner(self.delta_output, self.weights_array)) * activationf.backward()

        # update the weights
        for i in enumerate(self.weights_array):
            self.weights_array[i] = (self.weights_array[i] 
            + self.eta * delta * self.output_prevlayer[i])

        return float(delta)


class Layer:
    """ Class describing a single layer of the network
    """
    def __init__(self, number_units : int):
        self.number_units = number_units
        self.hidden_unit = HiddenUnit()
        self.output_unit = OutputUnit()
