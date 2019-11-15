import numpy as np
from tomomak.calc import multiply


class Mesh:
    """

    """

    def __init__(self, axes):
        self._axes = []
        self._dimension = 0
        for axis in axes:
            self.add_axis(axis)

    def __str__(self):
        res = "{}D mesh with {} axes:\n".format(self.dimension, len(self.axes))
        for i, ax in enumerate(self.axes):
            res += "{}. {} \n".format(i+1, ax)
        return res

    @property
    def dimension(self):
        return self._dimension

    @property
    def axes(self):
        return self._axes

    @property
    def shape(self):
        return tuple([ax.size for ax in self.axes])

    def add_axis(self, axis):
        self._axes.append(axis)
        self._dimension += axis.dimension

    def remove_axis(self, index=-1):
        self._dimension -= self._axes[index].dimension
        del self._axes[index]

    def integrate(self, data, index, integrate_type='integrate'):
        """ Calculates sum of data * dv or sum of data over given axes,
        where dv is length/surface/volume of a cell.

        Args:
            data (numpy.array):
            index (int/list of int):
            integrate_type (string):

        Returns:
            numpy.array: Integrated or summed array
        """
        if isinstance(index, int):
            index = [index]
        axis_shift = 0
        for i in index:
            axis = i + axis_shift
            if integrate_type != 'sum':
                dv = self.axes[i].volumes
                data = multiply.multiply_along_axis(data, dv, axis)
            data = np.sum(data, axis)
            axis_shift -= 1
        return data

    def _other(self, index):
        if isinstance(index, int):
            index = [index]
        invert_index = []
        for axis_index, _ in enumerate(self.axes):
            if all(axis_index != i for i in index):
                invert_index.append(axis_index)
        return invert_index

    def integrate_other(self, data, index):
        invert_index = self._other(index)
        return self.integrate(data, invert_index)

    def sum_other(self, data, index):
        invert_index = self._other(index)
        return self.integrate(data, invert_index, integrate_type='sum')

    def _prepare_data(self, data, index, data_type):
        if data_type == 'solution':
            new_data = self.integrate_other(data, index)
        elif data_type == 'detector_geometry':
            new_data = np.zeros([data.shape[0], data.shape[index+1]])
            for i, d in enumerate(data):
                new_data[i] = self.sum_other(d, index)
        else:
            raise AttributeError('data type {} is unknown'.format(data_type))
        return new_data

    def plot1d(self, data, index=0, data_type='solution', *args, **kwargs):
        new_data = self._prepare_data(data, index, data_type)
        plot, ax = self._axes[index].plot1d(new_data, data_type, *args, **kwargs)
        return plot, ax

    def plot2d(self, data, index=0, data_type='solution', *args, **kwargs):
        if isinstance(index, int):
            index = [index]
        # try to draw using 1 axis
        if len(index) == 1:
            try:
                new_data = self._prepare_data(data, index[0], data_type)
                plot, ax = self._axes[index[0]].plot2d(new_data, *args, **kwargs)
                return plot, ax
            except (AttributeError, TypeError):
                index.append(index[0] + 1)
        # try to draw using 2 axes
        new_data = self._prepare_data(data, index, data_type)
        try:
            plot, ax = self._axes[index[0]].plot2d(new_data, self._axes[index[1]], *args, **kwargs)
        except (AttributeError, NotImplementedError):
            new_data = new_data.transpose()
            plot, ax = self._axes[index[1]].plot2d(new_data, self._axes[index[0]], *args, **kwargs)
        return plot, ax

    def plot3d(self, data, index=0, *args, **kwargs):
        if isinstance(index, int):
            index = [index]
        # try to draw using 1 axis
        if isinstance(index, int):
            try:
                self._axes[index].plot3d(data, *args, **kwargs)
                return
            except AttributeError:
                index = [index, index+1]
        elif len(index) == 1:
            try:
                self._axes[index[0]].plot3d(data, *args, **kwargs)
                return
            except AttributeError:
                index.append(index[0] + 1)
        # try to draw using 2 axes
        try:
            axis2d = self._axes[index[0]].to2d(self._axes[index[1]])
        except AttributeError:
            axis2d = self._axes[index[0]].to3d(self._axes[index[1]])
        try:
            axis2d.plot3d(data, *args, **kwargs)
            return
        except (AttributeError, NotImplementedError):
            index.append(index[1] + 1)
        # try to draw using 3 axes
        axis3d = axis2d.to3d(self._axes[index[2]])
        axis3d.plot3d(data, *args, **kwargs)

    def draw_mesh(self):
        pass

    def density(self, data, coordinate):
        pass
