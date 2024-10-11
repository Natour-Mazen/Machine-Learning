import matplotlib.pyplot as plt
import numpy as np
from Perceptron_Simple import perceptron_simple, plot_with_class
from Plot_Display import plot_multiple

# f(x) = activation function
# f'(x) = 1 - f(x)^2
# new value = old value - learningRate * (-(targetValue - f(w * x) * f'(w * x) * x))
# newW = w - lr * (-(yd[0] - np.sign(w * x[0]) * derive(np.sign, w * x[0]) * x[0]))

DataSetName = ''


# Widrow-hof Learning.
def apprentissage_widrow(inputs, targets, epochs, batch_size):
    """
    Widrow-hof learning.

    :param inputs (x)  : Input values, mat[2, n]
    :param targets (yd) :  Result for the input values, vec[n].
    :param epochs: The number of loop on the training set.
    :param batch_size: The number of traited values before updating the weights.
    :return:
    """
    weights = np.random.randn(3)
    errors = []
    learning_rate = 0.1
    step = max(1, epochs // 10)
    plots_data = []
    plots_titles = []

    for epoch in range(epochs):
        temp_weights = weights.copy()
        epoch_error = 0

        for i, input_vec in enumerate(inputs):
            output = perceptron_simple(input_vec, weights, 1)
            error = - (targets[i] - output) * (1 - output * output)
            temp_weights += learning_rate * error * np.array([1, input_vec[0], input_vec[1]])
            epoch_error += error ** 2

            if (i + 1) % batch_size == 0:
                weights = temp_weights

        errors.append(epoch_error)
        print(f"Epoch {epoch + 1} : {epoch_error}")

        if epoch % step == 0:
            plots_data.append((inputs, weights, targets))
            plots_titles.append(f"{DataSetName} : Epoch {epoch + 1}")

        if epoch_error == 0 or (epoch > 0 and errors[epoch - 1] - epoch_error == 0):
            break

    plot_multiple(plots_data, plots_titles)
    return weights, errors


def run_widrow_learning(data_file, title, epochs, batch_size):
    """
    Run the Widrow-hoff learning.
    :param data_file: the data file
    :param title: title of the plot
    :param epochs: epochs number for the learning
    :param batch_size: batch size for the learning
    :return: None
    """
    global DataSetName
    data = np.loadtxt(data_file)
    targets = [1] * 25 + [-1] * 25
    plt.title(title)
    plt.scatter(data[0, :25], data[1, :25], c='r')
    plt.scatter(data[0, 25:], data[1, 25:], c='b')
    plt.legend(['Classe 1', 'Classe 2'])
    plt.show()

    DataSetName = title
    weights, errors = apprentissage_widrow(data.T, targets, epochs, batch_size)
    print("Weights : ", weights)

    plt.title(f"Errors {title} depending on the number of iterations")
    plt.plot(errors)
    plt.show()


if __name__ == '__main__':
    print("1.2.1 - Widrow-hoff learning programming")
    run_widrow_learning('ressources/p2_d1.txt', "1.2.1 Learning Widrow-hoff on Data_p2_d1", 500, 25)
    run_widrow_learning('ressources/p2_d2.txt', "1.2.1 Learning Widrow-hoff on Data_p2_d2", 500, 25)
