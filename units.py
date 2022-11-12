
"""Python code for the definition of classes
for different kinds of neurons in a neural network."""

import numpy as np

class OutputUnit:
    """Class for a general output unit."""
    def __init__(self, activation: np.ndarray, weights: np.ndarray,
                bias: float, eta: float):
        """Output unit initialization.

        Attributes
        ----------
        activation : function
            Activation function to be applied to net
            the current unit, so it is a one variable
            function defined outside the network.

        weights : arraylike of shape (n_components)
            Weights fot the current unit, taken from the
            weights matrix defined in the corrisponding layer.

        bias : float
            Treshold of the output unit.

        eta : float
            Learning rate for the alghoritm speed control.
        """
        # Saving in the unit the info initialized in the layer
        self.activation = activation
        self.weights = weights
        self.bias = bias
        self.eta = eta

        # Setting the initial numbers of itertion for the
        # backpropagation to zero (useful for changing eta)
        self.iters = 0

        # Defining some attributes useful for unit_forward methods
        self.net = None
        self.inputs = None
        self.output = None

        # Defining some attributes useful for unit_backword methods
        self.delta = None
        self.delta_weights = None

    def unit_forward(self, inputs: np.ndarray) -> float:
        """Method for the forward propagation of the output unit.

        Arguments
        ----------
        inputs : arraylike of shape (n_components)
            Set of data (output) from the units of the previous
            layer. The lenght of this vector is the fan in.

        Return
        ----------
        self.output : float
            Coputation of the unit output.
        """
        # Calculating the net and pass it through the activation function
        self.inputs = inputs
        self.net = np.dot(self.inputs, self.weights) + self.bias
        self.output = self.activation(self.net)
        return self.output

    def unit_backprop(self, error: float, fix_eta = True) -> float:
        """Method for the backpropagation of the output unit.

        Arguments
        ----------
        error : arraylike of shape (n_components)
            Difference between output and label of the output unit.

        fix_eta : bool
            Control parameter for eta, where False means change.

        Return
        ----------
        self.delta : float
            Small delta for the unit (delta_k).
        """
        # Determinatio of eta update rule
        if not fix_eta:
            eta = self.eta * np.exp( - self.iters / 100) # Exponential decay (example)
            self.iters += 1
        else:
            eta = self.eta

        # Backpropagation
        self.delta = error * self.activation(self.net)
        self.delta_weights = self.delta * self.inputs
        self.weights = self.weights + eta * self.delta_weights
        return self.delta
