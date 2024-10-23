import numpy as np
from matplotlib import pyplot as plt


def plot_with_class(data_points, weights, classes, title, min_y, max_y, space=0, font_size=10):
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
    plt.title(title, fontsize=font_size)
    plt.scatter(data_points[:, 0], data_points[:, 1], c=classes)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.xlim(min_y - space, max_y + space)
    plt.ylim(min_y - space, max_y + space)

    return plt


def plot_multiple(data_list, titles, max_plots_per_figure=4):
    """
    Plot multiple data sets with classification results.

    Parameters:
    data_list (list): A list of tuples, where each tuple contains:
                      - inputs (np.array): The input data points.
                      - weights (np.array): The synaptic weights of the neuron.
                      - targets (list): The classification results for each input.
    titles (list): A list of titles for each plot.

    Returns:
    None
    """

    num_plots = len(data_list)
    num_figures = (num_plots + max_plots_per_figure - 1) // max_plots_per_figure

    for fig_num in range(num_figures):
        plt.figure(figsize=(16, 19))  # Increase figure size
        start_idx = fig_num * max_plots_per_figure
        end_idx = min(start_idx + max_plots_per_figure, num_plots)

        for i in range(start_idx, end_idx):
            plt.subplot(2, 2, i - start_idx + 1)
            inputs, weights, targets = data_list[i]
            plot_with_class(inputs, weights, targets, titles[i], -1, 2, 3)
            plt.grid(True)

        plt.subplots_adjust(wspace=0.6, hspace=0.6)  # Increase spacing between subplots
        plt.show()


def plot_error_curve(cumulative_errors, data_file):
    plt.plot(cumulative_errors)
    plt.title(f"Error curve during training - {data_file}")
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Error")
    plt.show()



