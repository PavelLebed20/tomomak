from . import abstract_axes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Button
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
        self.size = size
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
        self.size = len(coordinates)
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

    def plot1d(self, data, data_type='solution', filled=True, fill_scheme=plt.cm.viridis, grid=False,  **kwargs):
        """

        :return:
        """
        if data_type == 'solution':
            fig, ax = plt.subplots()
            # n_bins = data.shape[0]
            # print(n_bins, data)
            # n, bins, patches = ax.hist(self.coordinates, weights=data, bins=n_bins, range=self.boundaries, **kwargs)
            color = None
            if filled:
                fracs = data / data.max()
                norm = colors.Normalize(fracs.min(), fracs.max())
                color = np.zeros((len(fracs), 4))
                for i, v in enumerate(fracs):
                    color[i] = fill_scheme(norm(v))
            ax.bar(self.coordinates, data, width=self.volumes, color=color, **kwargs)
            ax.set(xlabel="{}, {}".format(self.name, self.units),
                   ylabel=r"Density, {}{}".format(self.units, '$^{-1}$'),
                   title="Object density")
            if grid:
                ax.grid()
            plt.show()
        elif data_type == 'detector_geometry':

            freqs = np.arange(2, 20, 3)

            fig, ax = plt.subplots()
            plt.subplots_adjust(bottom=0.2)
            t = np.arange(0.0, 1.0, 0.001)
            s = np.sin(2 * np.pi * freqs[0] * t)
            l, = plt.plot(t, s, lw=2)

            class Index(object):
                ind = 0

                def next(self, event):
                    self.ind += 1
                    i = self.ind % len(freqs)
                    ydata = np.sin(2 * np.pi * freqs[i] * t)
                    l.set_ydata(ydata)
                    plt.draw()

                def prev(self, event):
                    self.ind -= 1
                    i = self.ind % len(freqs)
                    ydata = np.sin(2 * np.pi * freqs[i] * t)
                    l.set_ydata(ydata)
                    plt.draw()

            callback = Index()
            axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
            axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
            bnext = Button(axnext, 'Next')
            bnext.on_clicked(callback.next)
            bprev = Button(axprev, 'Previous')
            bprev.on_clicked(callback.prev)

            plt.show()
        else:
            raise AttributeError('data type {} is unknown'.format(data_type))

    def to2d(self, axis2):
        """

        :param axis2:
        :return:
        """
        pass
