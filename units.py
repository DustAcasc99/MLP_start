
"""Python code for the definition of classes
for different kinds of neurons in a neural network."""

import numpy as np

class OutputUnit:
    """Class for a general output unit."""
    def __init__(self, inputs: np.ndarray, activation: np.ndarray,
                weights: np.ndarray, eta: float):
        """Output unit initialization:

        Attributes
        ----------
        inputs : arraylike of shape (n_components)
            Values from the units of the previous
            layer in the neural network.

        activation : function
            Activation function to be applied to net
            the current unit, so it is a one variable
            function defined outside the network.

        weights : arraylike of shape (n_components)
            Weights fot the current unit.

        eta : float
            Learning rate for the alghoritm speed control.
        """
        # Should we define the weights directly inside the unit? (Andrea)
        # Should we pass the input only when forwardprop. is called? (Andrea)
        self.inputs = inputs
        self.activation = activation
        self.weights = weights # Maybe random generation also here like bias? (Andrea)
        self.eta = eta

        self.bias = np.random.uniform(-0.2, 0.2)

        self.net = None
        self.output = None
