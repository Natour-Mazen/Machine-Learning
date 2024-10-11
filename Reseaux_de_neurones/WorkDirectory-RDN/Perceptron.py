import numpy as np
import matplotlib.pyplot as plt


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


def plot_with_class(data_points, weights, classes, title, min_y, max_y):
    """
    Plot data points with classification results.

    Parameters:
    data_points (np.array): The input data points.
    weights (np.array): The synaptic weights of the neuron.
    classes (list): The classification results for each input.
    title (str): The title of the plot.
    min_y (float): The minimum y-axis value for the plot.
    max_y (float): The maximum y-axis value for the plot.

    Returns:
    plt: The plot object.
    """
    x_values = np.linspace(min_y, max_y)
    y_values = (weights[0] + x_values * weights[1]) / (-weights[2])

    plt.plot(x_values, y_values)
    plt.title(title)
    plt.scatter(data_points[:, 0], data_points[:, 1], c=classes)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.xlim(min_y, max_y)
    plt.ylim(min_y, max_y)

    return plt


if __name__ == '__main__':
    print("1.1 - Classification by simple perceptron on OR dataset")
    weights_OR = np.array([-0.5, 1, 1])
    the_data_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    results_OR = perceptron_simple(the_data_points, weights_OR, 0)
    print(results_OR)

    plot_with_class(
        the_data_points, weights_OR, results_OR,
        "1.1 - Classification by simple perceptron on OR dataset",
        -1, 2
    ).show()
