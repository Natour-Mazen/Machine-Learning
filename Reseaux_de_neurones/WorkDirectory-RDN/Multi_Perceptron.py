import numpy as np


def multi_layer_perceptron(input_vector, hidden_layer_weights, output_layer_weights):
    """
    Compute the output of a multi-layer perceptron.

    Parameters:
    input_vector (np.array) [x] : The input values (2 elements).
    hidden_layer_weights (np.array) [w1] : The synaptic weights of the first layer (3x2 matrix).
    output_layer_weights (np.array) [w2] : The synaptic weights of the second layer (3 elements).

    Returns:
    float: The output of the multi-layer perceptron.
    """

    def sigmoid_activation(x):
        return 1 / (1 + np.exp(-x))

    # Add bias term to the input
    input_with_bias = np.array([1, input_vector[0], input_vector[1]])

    # Compute the activations of the hidden layer
    hidden_neuron_1_input = np.dot(hidden_layer_weights[:, 0], input_with_bias)
    hidden_neuron_2_input = np.dot(hidden_layer_weights[:, 1], input_with_bias)

    hidden_neuron_1_output = sigmoid_activation(hidden_neuron_1_input)
    hidden_neuron_2_output = sigmoid_activation(hidden_neuron_2_input)

    # Add bias term to the hidden layer outputs
    hidden_layer_outputs_with_bias = np.array([1, hidden_neuron_1_output, hidden_neuron_2_output])

    # Compute the activation of the output layer
    output_neuron_input = np.dot(output_layer_weights, hidden_layer_outputs_with_bias)
    output_neuron_output = sigmoid_activation(output_neuron_input)

    return [hidden_neuron_1_output, hidden_neuron_2_output], output_neuron_output


if __name__ == '__main__':
    print("1.3.1 Mise en place d’un perceptron multicouche")

    # Test the multi-layer perceptron with the example input
    the_input_vector = np.array([1, 1])
    the_hidden_layer_weights = np.array([[-.5, .5], [2., .5], [-1., 1.]])
    the_output_layer_weights = np.array([2., -1., 1.])

    [hidden_neuron_1_output, hidden_neuron_2_output], output = multi_layer_perceptron(the_input_vector,
                                                                                      the_hidden_layer_weights,
                                                                                      the_output_layer_weights)
    print("Hidden Neuron 1 Output:", hidden_neuron_1_output)
    print("Hidden Neuron 2 Output:", hidden_neuron_2_output)
    print("Final Neuron Output:", output)
