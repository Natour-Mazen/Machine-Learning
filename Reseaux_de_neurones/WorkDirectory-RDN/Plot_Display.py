import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


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

        plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Increase spacing between subplots
        plt.show()


def plot_error_curve(cumulative_errors, data_file):
    plt.plot(cumulative_errors)
    plt.title(f"Error curve during training - {data_file}")
    plt.xlabel("Epoch")
    plt.ylabel("Cumulative Error")
    plt.show()


def display_loss_accuracy(name='', history=None, test_loss=None, test_acc=None):
    """
    Display training loss and accuracy curves.
    """
    loss = round(test_loss, 2)
    acc = round(test_acc * 100, 2)
    title_01 = f"Loss: {loss}"
    title_02 = f"Accuracy: {acc}"
    if name:
        title_01 = f"{name} - " + title_01
        title_02 = f"{name} - " + title_02
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title(title_01, fontsize=10)
    ax1.plot(history.history['loss'], 'r--', label='Loss of training data')
    ax1.plot(history.history['val_loss'], 'r', label='Loss of validation data')
    ax1.legend()
    ax2.set_title(title_02, fontsize=10)
    ax2.plot(history.history['accuracy'], 'g--', label='Accuracy of training data')
    ax2.plot(history.history['val_accuracy'], 'g', label='Accuracy of validation data')
    ax2.legend()
    plt.show()


def display_confusion_matrix(name='', Y_test=None, Y_predict=None, categories=None):
    """
    Display the confusion matrix of the model.
    """
    title = 'Confusion matrix of the model'
    if name:
        title = f"{name} - " + title
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(
            np.argmax(Y_test, axis=1),
            np.argmax(Y_predict, axis=1)
        ),
        display_labels=[c[0] for c in categories]
    ).plot()
    plt.title(title)
    plt.show()


