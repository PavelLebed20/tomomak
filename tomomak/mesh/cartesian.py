from . import abstract_axes
import numpy as np
import matplotlib.pyplot as plt
from tomomak.plots import plot1d, plot2d
import warnings


class Axis1d(abstract_axes.Abstract1dAxis):
    def __init__(self, coordinates=None, lower_limit=0, upper_limit=None, size=None, name="", units=""):
        super().__init__(name, units)
        if coordinates is None:
            if size is None:
                warnings.warn("Axis1d init: size was not set. Default size = 10 is used.")
                size = 10
            if upper_limit is None:
                warnings.warn("Axis1d init: upper_limit  was not set. Default upper_limit = 10 is used.")
                upper_limit = 10
            self._create_using_limits(lower_limit, upper_limit, size)
        else:
            if size is not None or upper_limit is not None:
                warnings.warn("Since coordinates are given explicitly, size and upper_limit arguments are ignored.")
            self._create_using_coordinates(coordinates, lower_limit)

    def _create_using_limits(self, lower_limit, upper_limit, size):
        self._size = size
        self._boundaries = [lower_limit, upper_limit]
        dv = np.abs(upper_limit - lower_limit) / size
        self._volumes = np.full(size, dv)
        self._coordinates = np.fromfunction(lambda i: lower_limit + (i * dv) + (dv / 2), (size,))
        self._calc_cell_edges()

    def _create_using_coordinates(self, coordinates, lower_limit):
        if (any(np.diff(coordinates)) < 0 and coordinates[-1] > lower_limit
                or any(np.diff(coordinates)) > 0 and coordinates[-1] < lower_limit):
            raise Exception("Coordinates are not monotonous.")
        if (coordinates[-1] > lower_limit > coordinates[0]
                or coordinates[-1] < lower_limit < coordinates[0]):
            raise Exception("lower_limit is inside of the first segment.")
        self._size = len(coordinates)
        self._coordinates = coordinates
        dv = np.diff(coordinates)
        self._volumes = np.zeros(self.size)
        dv0 = coordinates[0] - lower_limit
        self._volumes[0] = 2 * dv0
        for i in range(self.size - 1):
            self._volumes[i + 1] = 2 * dv[i] - self._volumes[i]
        for i, v in enumerate(self._volumes):
            if v <= 0:
                raise Exception("{} point of the coordinates is inside of previous segment. "
                                "Increase the distance between the points.".format(i))
        upper_limit = coordinates[-1] + self.volumes[-1] / 2
        self._boundaries = [lower_limit, upper_limit]
        self._calc_cell_edges()

    def _calc_cell_edges(self):
        self._cell_edges = np.zeros(self.size + 1)
        self._cell_edges[0] = self.boundaries[0]
        for i in range(self.size):
            self._cell_edges[i + 1] = self._volumes[i] + self._cell_edges[i]
        return self._cell_edges

    def __str__(self):
        if self.regular:
            ax_type = 'regular'
        else:
            ax_type = 'irregular'
        return "{}D {} axis with {} cells. Name: {}. Boundaries: {} {}. "\
            .format(self.dimension, ax_type, self.size, self.name, self._boundaries, self.units)

    @property
    def volumes(self):
        return self._volumes

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def boundaries(self):
        return self._boundaries

    @property
    def cell_edges(self):
        return self._cell_edges

    @property
    def size(self):
        return self._size

    @property
    def regular(self):
        if all(self._volumes - self._volumes[0] == 0):
            return True
        else:
            return False

    def plot1d(self, data, data_type='solution', filled=True,
               fill_scheme='viridis', edgecolor='black', grid=False, equal_norm=False, *args, **kwargs):
        """

        :return:
        """
        if data_type == 'solution':
            y_label = r"Density, {}{}".format(self.units, '$^{-1}$')
            plot, ax = plot1d.bar1d(data, self, 'Density', y_label, filled, fill_scheme,
                                    edgecolor, grid, *args, **kwargs)
        elif data_type == 'detector_geometry':
            title = 'Detector 1/{}'.format(data.shape[0])
            y_label = 'Intersection length, {}'.format(self.units)
            plot, ax, _ = plot1d.detector_bar1d(data, self, title, y_label, filled,
                                                fill_scheme, edgecolor, grid, equal_norm, *args, **kwargs)
        else:
            raise AttributeError('data type {} is unknown'.format(data_type))
        plt.show()
        return plot, ax

    def plot2d(self, data, axis2, data_type='solution',
               fill_scheme='viridis',  grid=False, equal_norm=False, *args, **kwargs):
        """

        """
        if type(axis2) is not Axis1d:
            raise NotImplementedError("2D plots with such combination of axes are not supported.")
        if data_type == 'solution':
            title = r"Density, {}{}{}{}".format(self.units, '$^{-1}$', axis2.units, '$^{-1}$')
            plot, ax, fig, cb = plot2d.colormesh2d(data, self, axis2, title, fill_scheme, grid, *args,  **kwargs)
        elif data_type == 'detector_geometry':
            title = 'Detector 1/{}'.format(data.shape[0])
            plot, ax, _ = plot2d.detector_colormesh2d(data, self, axis2, title, fill_scheme, grid,
                                                      equal_norm, *args, **kwargs)
        else:
            raise AttributeError('data type {} is unknown'.format(data_type))
        plt.show()
        return plot, ax
