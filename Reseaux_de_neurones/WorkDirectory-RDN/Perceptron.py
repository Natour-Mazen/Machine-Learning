import numpy as np


def perceptron_simple(x: np.array, w: np.array, active: int) -> list:
    """
     Perceptron simple.
    :param x: Synaptic weights of the neuron, vec3.
    :param w: The input of the neural network
    :param active: Tell if the
    :return:
    """
    seuil : float = w[0]
    dot_product : float = np.dot(x, w[1:])
    res : float = dot_product + seuil
    return np.sign(res) if (active == 0) else np.tanh(res)