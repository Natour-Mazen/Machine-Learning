import numpy as np
from matplotlib import pyplot as plt
from Multi_Perceptron import multi_layer_perceptron
from Plot_Display import plot_error_curve


def sigmoid_derivative(x):
    return x * (1 - x)


def update_weights(weights, delta_weights, learning_rate):
    return weights + learning_rate * delta_weights


def multiperceptron_widrow(inputs, targets, epochs, batch_size, learning_rate=0.5):
    weights_hidden = np.random.rand(3, 2) - 0.5
    weights_output = np.random.rand(3) - 0.5
    delta_weights_hidden = np.zeros((3, 2))
    delta_weights_output = np.zeros(3)
    cumulative_errors = np.zeros(epochs)

    for epoch in range(epochs):
        for sample_index in range(inputs.shape[1]):
            input_sample = inputs[:, sample_index]
            target = targets[sample_index]

            hidden_layer_output, final_output = multi_layer_perceptron(input_sample, weights_hidden, weights_output)
            cumulative_errors[epoch] += (target - final_output) ** 2

            output_error = -(target - final_output) * sigmoid_derivative(final_output)
            hidden_error_1 = weights_output[1] * output_error * sigmoid_derivative(hidden_layer_output[0])
            hidden_error_2 = weights_output[2] * output_error * sigmoid_derivative(hidden_layer_output[1])

            input_with_bias = np.array([1, input_sample[0], input_sample[1]])
            delta_weights_hidden[:, 0] += -learning_rate * hidden_error_1 * input_with_bias
            delta_weights_hidden[:, 1] += -learning_rate * hidden_error_2 * input_with_bias

            hidden_output_with_bias = np.array([1, hidden_layer_output[0], hidden_layer_output[1]])
            delta_weights_output += -learning_rate * output_error * hidden_output_with_bias

            if sample_index % batch_size == 0:
                weights_hidden = update_weights(weights_hidden, delta_weights_hidden, learning_rate)
                weights_output = update_weights(weights_output, delta_weights_output, learning_rate)
                delta_weights_hidden = np.zeros((3, 2))
                delta_weights_output = np.zeros(3)

        error = round(cumulative_errors[epoch], 3)
        #print(f'Epoch {epoch + 1}/{epochs} - Error: {error}')
        if error < 0.01:
            break

    return weights_hidden, weights_output, cumulative_errors


def plot_training_set(inputs, targets):
    plt.scatter(inputs[0, targets == 0], inputs[1, targets == 0], c='r')
    plt.scatter(inputs[0, targets == 1], inputs[1, targets == 1], c='b')
    plt.legend(['False', 'True'])
    plt.title("XOR Training Set")
    plt.show()


def get_prediction(input_coords, weights_hidden, weights_output):
    return round(multi_layer_perceptron(np.array(input_coords), weights_hidden, weights_output)[1])


def plot_decision_boundary(weights_hidden, weights_output, data_points):
    x = np.linspace(-2, 2)
    y1 = (weights_hidden[0, 0] + x * weights_hidden[1, 0]) / (-weights_hidden[2, 0])
    y2 = (weights_hidden[0, 1] + x * weights_hidden[1, 1]) / (-weights_hidden[2, 1])
    y3 = (weights_output[0] + x * weights_output[1]) / (-weights_output[2])

    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.scatter(data_points[:, 0], data_points[:, 1])
    plt.legend(['Hidden Neuron 1', 'Hidden Neuron 2', 'Output Neuron', 'Data Points'])
    plt.title("Decision Boundary Plot")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    inputs = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    targets = np.array([0, 1, 1, 0])

    plot_training_set(inputs, targets)

    the_weights_hidden, the_weights_output, the_cumulative_errors = multiperceptron_widrow(inputs, targets, 20000, 4)

    print('x = [0, 0] | y =', get_prediction([0, 0], the_weights_hidden, the_weights_output))
    print('x = [0, 1] | y =', get_prediction([0, 1], the_weights_hidden, the_weights_output))
    print('x = [1, 0] | y =', get_prediction([1, 0], the_weights_hidden, the_weights_output))
    print('x = [1, 1] | y =', get_prediction([1, 1], the_weights_hidden, the_weights_output))

    plot_error_curve(the_cumulative_errors, "XOR")
    plot_decision_boundary(the_weights_hidden, the_weights_output, np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
