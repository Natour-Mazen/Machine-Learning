import matplotlib.pyplot as plt
import numpy as np
from Perceptron_Simple import perceptron_simple, plot_with_class
from Plot_Display import plot_multiple, plot_error_curve

DataSetName = ''


def tanhDerive(value: float) -> float:
    """
    The derivative of the tanh function.
    :param value: The value to evaluate.
    :return: The result of the derivative.
    """
    return 1 - np.tanh(value) ** 2


# Widrow-hof Learning.
def apprentissage_widrow(inputs, targets, epochs, batch_size) -> (list, list):
    """
    Widrow-hof learning.

    :param inputs (x)  : Input values, mat[2, n]
    :param targets (yd) :  Result for the input values, vec[n].
    :param epochs: The number of loop on the training set.
    :param batch_size: The number of traited values before updating the weights.
    :return: The list of the final weights and the list of the errors.
    """
    weights = np.random.randn(3)
    errors = []
    learning_rate = 0.1
    step = max(1, epochs // 10)
    plots_data = []
    plots_titles = []

    for epoch in range(epochs):
        temp_weights = weights.copy()
        epoch_error: float = 0.

        # Combined inputs and targets into tuples.
        combined_inputs_targets = list(zip(inputs, targets))

        # We shuffle the combined list to have better results.
        np.random.shuffle(combined_inputs_targets)

        # We unzip the combined list back into inputs and targets.
        shuffle_inputs, shuffle_targets = zip(*combined_inputs_targets)

        # Convert them back to array.
        shuffle_inputs = np.array(inputs)
        shuffle_targets = np.array(targets)

        # For all the elements, in input_vec we have an array with 2 values : [v1, v2].
        for i, input_vec in enumerate(shuffle_inputs):
            reel_output: float = perceptron_simple(input_vec, weights, 1)

            # We transform the output between -1 and 1 in just -1 or 1.
            output = np.sign(reel_output)

            # We insert 1. at the start of input vector because we need a value to calculate the first weight.
            input_vec = np.insert(input_vec, 0, 1.)

            # The value pass to the activation function.
            value_activation: float = np.dot(weights, input_vec)
            # Value through the tanh function.
            tanh_result: float = reel_output
            # Value through the derivative of the tanh function.
            tanh_derive_result: float = tanhDerive(value_activation)

            # Formula to have the new weight :
            # newW = w - lr * (-(yd[i] - tanh(w * x) * tanh_derive(w * x) * x[i]))

            # Update the weights.
            temp_weights[0] = temp_weights[0] - learning_rate * (
                        -(shuffle_targets[i] - tanh_result) * tanh_derive_result * input_vec[0])
            temp_weights[1] = temp_weights[1] - learning_rate * (
                        -(shuffle_targets[i] - tanh_result) * tanh_derive_result * input_vec[1])
            temp_weights[2] = temp_weights[2] - learning_rate * (
                        -(shuffle_targets[i] - tanh_result) * tanh_derive_result * input_vec[2])

            # We calculate the error.
            epoch_error += (shuffle_targets[i] - output) ** 2

            # If we have process 'batch_size' elements, we change the weights.
            if (i + 1) % batch_size == 0:
                weights = temp_weights

        errors.append(epoch_error)
        print(f"Epoch {epoch + 1} : {epoch_error}")

        if epoch % step == 0:
            plots_data.append((shuffle_inputs, weights, shuffle_targets))
            plots_titles.append(f"{DataSetName} : Epoch {epoch + 1}, Error {epoch_error}")

        # If we have an error egal to 0, we stop because it can't be better.
        if epoch_error == 0.:
            break

    plot_multiple(plots_data, plots_titles)
    return weights, errors


def run_widrow_learning(data_file, title, epochs, batch_size) -> (int, float):
    """
    Run the Widrow-hoff learning.
    :param data_file: the data file
    :param title: title of the plot
    :param epochs: epochs number for the learning
    :param batch_size: batch size for the learning
    :return: The number of errors to have the best solution and the best error obtained.
    """
    global DataSetName
    data = np.loadtxt(data_file)
    targets = [1] * 25 + [-1] * 25
    title_points = f"Distribution of points - {data_file}"
    plt.title(title_points)
    plt.scatter(data[0, :25], data[1, :25], c='r')
    plt.scatter(data[0, 25:], data[1, 25:], c='b')
    plt.legend(['Classe 1', 'Classe 2'])
    plt.show()

    DataSetName = title
    weights, errors = apprentissage_widrow(data.T, targets, epochs, batch_size)
    print("Weights : ", weights)

    plot_error_curve(errors, data_file)

    #plt.title(f"Errors depending on the number of epochs - {data_file}")
    #plt.plot(errors)
    #plt.show()

    return len(errors), errors[-1]


if __name__ == '__main__':
    print("1.2.1 - Widrow-hoff learning programming")

    # To see in average the number of epoch to have the best solution.

    avgError: int = 0
    bestErrorAvg: float = 0.
    size : int = 100
    for i in range(size):
        nb_error, best_error = run_widrow_learning('ressources/p2_d1.txt', "1.2.1 - Widrow on Data_p2_d1", 10, 1)
        avgError += nb_error
        bestErrorAvg += best_error
    print("avgError: ", avgError / size)
    print("bestErrorAvg: ", bestErrorAvg / size)
    # ~4 epoch to have the best solution.

    # To see in average the best lost that we can get.

    # avgError : int = 0
    # bestErrorAvg : float = 0.
    # size: int = 10
    # for i in range(size):
    #     nb_error, best_error = run_widrow_learning('ressources/p2_d2.txt', "1.2.1 - Widrow on Data_p2_d2", 50, 8)
    #     avgError += nb_error
    #     bestErrorAvg += best_error
    # print("avgError: ", avgError / size)
    # print("bestErrorAvg: ", bestErrorAvg / size)
    # ~50 epochs to have the best solution (error at 8).

    run_widrow_learning('ressources/p2_d1.txt', "1.2.1 - Widrow on Data_p2_d1", 10, 8)
    run_widrow_learning('ressources/p2_d2.txt', "1.2.1 - Widrow on Data_p2_d2", 50, 8)

