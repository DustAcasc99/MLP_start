
""" Test for the classes in topologyNN.py for the neural network.
"""

# import packages and classes
import unittest
import numpy as np
from neuralnetwork.topologyNN import OutputUnit, HiddenUnit, OutputLayer, HiddenLayer

# Tests for a linear activation function

# defines a linear activation function
def linear(net):

    """ Function simply returning the input.
    """

    return net


# tests for the units classes
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

        # defining the target objects that we need to test
        self.target_outunit = OutputUnit(linear, self.weights, self.bias, self.eta)
        self.target_hidunit = HiddenUnit(linear, self.weights, self.bias, self.eta)

    def test_contructor(self):

        """ Test for the basic attributes of the unit's contructor.
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

        """ Test for the feedforward method for units.
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

        """ Test for the backpropagation methods for units.
        """

        # let's see what happens to weights during backpropagation for the output unit
        print(f'Weights array before backpropagation of OutputUnit: \
                \n {self.target_outunit.weights_array}')
        self.target_outunit.feedforward_unit(self.inputs) # initialize the attributes
        delta_k = (self.target_outunit.backprop_unit(self.label))
        print(f'Weights array after backpropagation of OutputUnit (should decrease): \
                \n {self.target_outunit.weights_array}')
        print(f'The computed delta: {delta_k} \n')

        # let's see what appens to weights during backpropagation for the hidden unit
        print(f'Weights array before backpropagation of HiddenUnit: \
                \n {self.target_hidunit.weights_array}')
        self.target_hidunit.feedforward_unit(self.inputs) # initialize the attributes
        delta_j = (self.target_hidunit.backprop_unit(delta_k, self.target_outunit.weights_array[1])) # j=1
        print(f'Weights array after backpropagation of HiddenUnit (should increase): \
                \n {self.target_hidunit.weights_array}')
        print(f'The computed delta: {delta_j} \n')

        # check the effect of one input with a minibatch > 1
        self.target_hidunit.weights_array = self.weights
        delta_j = (self.target_hidunit.backprop_unit(delta_k,
                    self.target_outunit.weights_array[1], minibatch_size=2))
        print(f'Weights array after backpropagation 1/2 of HiddenUnit (should be the same): \
                \n {self.target_hidunit.weights_array}')
        print(f'The computed delta: {delta_j} \n')
        delta_j = (self.target_hidunit.backprop_unit(delta_k,
                    self.target_outunit.weights_array[1], minibatch_size=2))
        print(f'Weights array after backpropagation 2/2 of HiddenUnit (should change): \
                \n {self.target_hidunit.weights_array}')
        print(f'The computed delta: {delta_j} \n')

        # check the effect of momentum
        self.target_hidunit.weights_array = self.weights
        old = self.target_hidunit.old_weight_change
        delta_j = (self.target_hidunit.backprop_unit(delta_k,
                    self.target_outunit.weights_array[1]))
        print(f'Weights array after backpropagation of HiddenUnit without alpha: \
                \n {self.target_hidunit.weights_array}')
        print(f'The computed delta: {delta_j} \n')
        self.target_hidunit.weights_array = self.weights
        self.target_hidunit.alpha = 0.5
        self.target_hidunit.old_weight_change = old
        delta_j = (self.target_hidunit.backprop_unit(delta_k,
                    self.target_outunit.weights_array[1]))
        print(f'Weights array after backpropagation of HiddenUnit with alpha: \
                \n {self.target_hidunit.weights_array}')
        print(f'The computed delta: {delta_j} \n')

# tests for the layers classes
class TestNetworkLayers(unittest.TestCase):

    """ Class containing a test for every functionality
    ot the classes OutputLayer and HiddenLayer.
    """
    def setUp(self, fan_in = 5):

        """ Setup information for the layer.
        """

        # defining some data and some weights
        self.number_units = 6
        self.inputs = np.ones(fan_in)
        self.eta = 1.

        # target values for the units of the output layer
        self.target_olayer = np.full((self.number_units), 3)
        # note: we're considering the number of units of the first
        # outer layer equal to that of the layer at hand + 1
        self.delta_next = np.full((self.number_units), 0.5)
        self.weights_matrix_next = np.ones((self.number_units + 1, self.number_units))

        # defining the target objects that we need to test
        self.target_outlayer = OutputLayer(linear, self.number_units, self.inputs, self.eta)
        self.target_hidlayer = HiddenLayer(linear, self.number_units, self.inputs, self.eta)

        # initializing the weights matrix and the bias array values
        self.target_outlayer.weights_matrix = np.ones((self.number_units, fan_in))
        self.target_hidlayer.weights_matrix = np.ones((self.number_units, fan_in))

        self.target_outlayer.bias_array = np.zeros(self.number_units)
        self.target_hidlayer.bias_array = np.zeros(self.number_units)


    def test_layercontructor(self):

        """ Test for attributes of the output layer's contructor.
        """

        # checking the correct load of data in the contructor
        self.assertTrue(self.target_outlayer.eta == self.eta)
        self.assertTrue(self.target_outlayer.number_units == self.number_units)
        self.assertTrue((self.target_outlayer.inputs == self.inputs).all())

       # checking the equality between output layer and hidden layer contructor
        self.assertTrue(self.target_outlayer.eta == self.target_hidlayer.eta)
        self.assertTrue(self.target_outlayer.number_units == self.target_hidlayer.number_units)
        self.assertTrue((self.target_outlayer.inputs == self.target_hidlayer.inputs).all())


    def test_layerfeedforward(self):

        """ Test for the feedforward method for layers.
        """

        # test output layer computation
        computed = self.target_outlayer.feedforward_layer()
        expected = (np.inner(self.inputs, self.target_outlayer.weights_matrix)
                   + self.target_outlayer.bias_array)
        self.assertAlmostEqual(computed.all(), expected.all())

        # test hidden layer computation
        computed = self.target_hidlayer.feedforward_layer()
        expected = (np.inner(self.inputs, self.target_hidlayer.weights_matrix)
                   + self.target_hidlayer.bias_array)
        self.assertAlmostEqual(computed.all(), expected.all())


    def test_layerbackprop(self):

        """ Test for the backpropagation methods for layers.
        """

        # for the output layer:

        # print the weights matrix before the backpropagation
        print(f'The weights matrix before the backpropagation of the OutputLayer: \
               \n {self.target_outlayer.weights_matrix}')

        # output of every layer's unit after the feedforward propagation
        layer_outoutput = self.target_outlayer.feedforward_layer()

        computed = self.target_outlayer.backprop_layer(self.target_olayer)
        expected = (self.target_olayer - layer_outoutput)
        self.assertAlmostEqual(computed.all(), expected.all())

        # print the weights matrix before the backpropagation
        print(f'The weights matrix after the backpropagation of the OutputLayer: \
               \n {self.target_outlayer.weights_matrix}')

        # for the hidden layer:

        # print the weights matrix before the backpropagation
        print(f'The weights matrix before the backpropagation of the HiddenLayer: \
               \n {self.target_hidlayer.weights_matrix}')

        layer_outoutput = self.target_hidlayer.feedforward_layer()

        computed = self.target_hidlayer.backprop_layer(self.delta_next, self.weights_matrix_next)
        expected = np.inner(self.delta_next, self.weights_matrix_next)
        self.assertAlmostEqual(computed.all(), expected.all())

        # print the weights matrix before the backpropagation
        print(f'The weights matrix after the backpropagation of the HiddenLayer: \
               \n {self.target_hidlayer.weights_matrix}')



if __name__ == "__main__":
    unittest.main()
