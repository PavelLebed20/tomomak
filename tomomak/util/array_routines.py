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


def broadcast_object(ar, index, shape):
    if isinstance(index, int):
        index = [index]
    if ar.shape == shape:
        return ar
    else:
        # Moving current axes to the end in order to prepare shape for numpy broadcast
        shape = list(shape)
        index = list(index)
        index.reverse()
        current_shape = []
        for i, ind in enumerate(index):
            val = shape.pop(ind)
            current_shape.append(val)
        current_shape.reverse()
        for ind in current_shape:
            shape.append(ind)
        # broadcasting
        ar = np.broadcast_to(ar, shape)
        index.reverse()
        # making correct axes order
        ind_len = len(index)
        for i, ind in enumerate(index):
            ar = np.moveaxis(ar, -ind_len + i, ind)
        return ar



