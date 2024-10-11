import numpy as np


# f(x) = activation function
# f'(x) = 1 - f(x)^2
# new value = old value - learningRate * (-(targetValue - f(w * x) * f'(w * x) * x))

# Widrow-hof Learning.
def apprentissage_widrow(x: np.ndarray, yd: np.array, epoch: int, batch_size: int) -> (np.array, np.array):
    """
    Widrow-hof learning.

    :param x: Input values, mat[2, n]
    :param yd: Result for the input values, vec[n].
    :param epoch: The number of loop on the training set.
    :param batch_size: The number of traited values before updating the weights.
    :return:
    """
    w = np.random.rand(3)
    errors = []


    pass


if __name__ == '__main__':
    print("1.2.1 - Widrow-hoff learning programming")
