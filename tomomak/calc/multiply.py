import numpy as np


def multiply_along_axis(a, b, axis):
    """ Multiply numpy array by 1-D numpy array along given axis.

    Args:
        a(ndarray): Array to multiply.
        b(ndarray): 1D  array to multiply by.
        axis(int): Axis along which multiplication is performed.

    Returns:
        ndarray: Multiplied array
    """
    dim_array = np.ones((1, a.ndim), int).ravel()
    dim_array[axis] = -1
    b_reshaped = b.reshape(dim_array)
    return a * b_reshaped
