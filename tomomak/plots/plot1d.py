import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Button


def bar1d(data, axis, title='', ylabel='', filled=True, fill_scheme=plt.cm.viridis, grid=False, **kwargs):
    fig, ax = plt.subplots()
    color = None
    if filled:
        data_norm = data / data.max()
        norm = colors.Normalize(data_norm.min(), data_norm.max())
        color = np.zeros((len(data_norm), 4))
        for i, v in enumerate(data_norm):
            color[i] = fill_scheme(norm(v))
    plot = ax.bar(axis.coordinates, data, width=axis.volumes, color=color, **kwargs)
    ax.set(xlabel="{}, {}".format(axis.name, axis.units),
           ylabel=ylabel, title=title)
    if grid:
        ax.grid()
    return plot, ax


def detector_bar1d(data, axis, title='', ylabel='', filled=True, fill_scheme=plt.cm.viridis, grid=False, **kwargs):
    class Index(object):
        ind = 0

        def next(self, _):
            self.ind += 1
            i = self.ind % data.shape[0]
            y_data = data[i]
            for r, h in zip(plot, y_data):
                r.set_height(h)
            new_title = 'Detector {}/{}'.format(i + 1, data.shape[0])
            ax.set(title=new_title.format(data.shape[0]))
            plt.draw()

        def prev(self, _):
            self.ind -= 1
            i = self.ind % data.shape[0]
            y_data = data[i]
            for r, h in zip(plot, y_data):
                r.set_height(h)
            new_title = 'Detector {}/{}'.format(i + 1, data.shape[0])
            ax.set(title=new_title.format(data.shape[0]))
            plt.draw()

    plot, ax = bar1d(data[0], axis, title=title, ylabel=ylabel,
                     filled=filled, fill_scheme=fill_scheme, grid=grid, **kwargs)
    plt.subplots_adjust(bottom=0.2)
    callback = Index()
    ax_prev = plt.axes([0.7, 0.02, 0.1, 0.075])
    ax_next = plt.axes([0.81, 0.02, 0.1, 0.075])
    b_next = Button(ax_next, 'Next')
    b_next.on_clicked(callback.next)
    b_prev = Button(ax_prev, 'Previous')
    b_prev.on_clicked(callback.prev)
    return plot, ax, (b_next, b_prev)
