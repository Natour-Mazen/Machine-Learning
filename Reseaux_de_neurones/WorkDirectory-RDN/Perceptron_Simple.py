import numpy as np
from Plot_Display import plot_with_class


# Simple Perceptron.
def perceptron_simple(inputs: np.array, weights: np.array, activation_function: int) -> list:
    """
    Simple Perceptron.

    Parameters:
    inputs (np.array): The input data points.
    weights (np.array): The synaptic weights of the neuron.
    activation_function (int): Selector for the activation function (0 for sign, 1 for tanh).

    Returns:
    list: The activation result for each input.
    """
    threshold: float = weights[0]
    dot_product: float = np.dot(inputs, weights[1:])
    result: float = dot_product + threshold
    return np.sign(result) if (activation_function == 0) else np.tanh(result)


if __name__ == '__main__':
    print("1.1 - Classification by simple perceptron on OR dataset")
    weights_OR = np.array([-0.5, 1, 1])
    the_data_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    results_OR = perceptron_simple(the_data_points, weights_OR, 0)
    print("results_OR : ", results_OR)

    plot_with_class(
        the_data_points, weights_OR, results_OR,
        "1.1 - Classification by simple perceptron on OR dataset",
        -1, 2
    ).show()
