import numpy as np


def perceptron_simple(x: list, w: np.ndarray, active: int) -> list:
    """
     Perceptron simple.
    :param x: Synaptic weights of the neuron, vec3.
    :param w: The input of the neural network
    :param active:
    :return:
    """
    seuil = w[0]
    dot_product = np.dot(x, w[1:])
    res = dot_product + seuil
    return np.sign(res) if (active == 0) else np.tanh(res)