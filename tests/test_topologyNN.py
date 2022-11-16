
""" Test for the classes in topologyNN.py for the neural network.
"""

# import packages and classes
import unittest
import numpy as np
from neuralnetwork.topologyNN import OutputUnit, HiddenUnit

# defining a linear activation function
def linear(net):

    """ Function simply returning the input.
    """

    return net

class TestNetworkUnits(unittest.TestCase):

    """ Class containing a test for every functionality
    ot the classes OutputUnit and HiddenUnit.
    """

    def setUp(self, fan_in = 5):

        """ Setup information for the neurons.
        """

        # defining some data and some weights (for a single neuron)
        self.inputs = np.ones(fan_in)
        self.weights = np.ones(fan_in)
        self.bias = 0.
        self.label = 1.
        self.eta = 1.

        # defining the target objects
        self.target_outunit = OutputUnit(linear, self.weights, self.bias, self.eta)
        self.target_hidunit = HiddenUnit(linear, self.weights, self.bias, self.eta)

    def test_contructor(self):

        """ Test for the basic attriutes of unit's contructor.
        """

        # checking the correct load of data in the contructor
        self.assertTrue((self.target_hidunit.weights_array == self.weights).all())
        self.assertTrue(self.target_hidunit.bias == self.bias)
        self.assertTrue(self.target_hidunit.eta == self.eta)

        # checking the equality between output unit and hidden unit contructor
        self.assertTrue((self.target_hidunit.weights_array
                         == self.target_outunit.weights_array).all())
        self.assertTrue(self.target_hidunit.bias == self.target_outunit.bias)
        self.assertTrue(self.target_hidunit.eta == self.target_outunit.eta)

    def test_feedforward(self):

        """ Test for the feedforward mthod for units.
        """

        # test output unit computation
        computed = self.target_outunit.feedforward_unit(self.inputs)
        expected = np.inner(self.inputs, self.weights) + self.bias
        self.assertEqual(computed, expected)

        # test hidden unit computation
        computed = self.target_hidunit.feedforward_unit(self.inputs)
        expected = np.inner(self.inputs, self.weights) + self.bias
        self.assertEqual(computed, expected)

    def test_backprop_unit(self):

        """ Test for the backporpagation methods for units.
        """

        # let's see what appens to weights during backpropagation for output unit
        print(f'Weights array before backpropagation of OutputUnit: \
                \n {self.target_outunit.weights_array}')
        self.target_outunit.feedforward_unit(self.inputs) # initialize the attributes
        delta_k = self.target_outunit.backprop_unit(self.label)
        print(f'Weights array after backpropagation of OutputUnit (should be decreased): \
                \n {self.target_outunit.weights_array}')
        print(f'The computed delta: {delta_k} \n')

        # let's see what appens to weights during backpropagation for hidden unit
        print(f'Weights array before backpropagation of HiddenUnit: \
                \n {self.target_hidunit.weights_array}')
        self.target_hidunit.feedforward_unit(self.inputs) # initialize the attributes
        delta_j = self.target_hidunit.backprop_unit(delta_k, self.target_outunit.weights_array[1]) # j=1
        print(f'Weights array after backpropagation of HiddenUnit (should increase): \
                \n {self.target_hidunit.weights_array}')
        print(f'The computed delta: {delta_j} \n')

if __name__ == "__main__":
    unittest.main()
