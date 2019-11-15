import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Button
from . import interactive


def bar1d(data, axis, title='', ylabel='', filled=True, fill_scheme='viridis', edgecolor='black',
          grid=False, *args,  **kwargs):
    """Prepare bar plot for 1D data visualization.
    matplotlib.pyplot.bar plot is used.

    Args:
        data(ndarray): 1D array of data.
        axis(axis): corresponding tomomak axis.
        title(str, optional): Plot title. default: ''.
        ylabel(str, optional): Plot y axis label. : ''.
        filled(bool, optional): if True, bars are filled with colors from fill_scheme according to their values.
        fill_scheme(pyplot colormap): pyplot colormap for to be used if filled is true or one color.
        grid(bool): if True, grid is shown.
        *args, **kwargs: arguments will be passed to matplotlib.pyplot.bar

    Returns:
        plot: matplotlib bar plot.
        ax(axes.Axes ): axes.Axes object or array of Axes objects.
            See matplotlib Axes class
    """
    fig, ax = plt.subplots()
    color = None
    if filled:
        color = _fill_plot(data, fill_scheme)
    ax.set(xlabel="{}, {}".format(axis.name, axis.units),
           ylabel=ylabel, title=title)
    if grid:
        ax.grid()
    plot = ax.bar(axis.coordinates, data, width=axis.volumes, color=color, edgecolor=edgecolor, *args, **kwargs)
    return plot, ax


def _fill_plot(data, fill_scheme):
    data_norm = data / data.max()
    norm = colors.Normalize(data_norm.min(), data_norm.max())
    color = np.zeros((len(data_norm), 4))
    fill_scheme = plt.get_cmap(fill_scheme)
    for i, v in enumerate(data_norm):
        color[i] = fill_scheme(norm(v))
    return color

def detector_bar1d(data, axis, title='', ylabel='', filled=True,
                   fill_scheme='viridis', edgecolor='black', grid=False, *args, **kwargs):

    class BarPlotSlice(interactive.DetectorPlotSlicer):
        def redraw(self):

            y_data = self.data[self.ind]
            for r, h in zip(plot, y_data):
                r.set_height(h)
            color = _fill_plot(y_data, fill_scheme)
            for c, p in zip(color, plot):
                if filled:
                    p.set_color(c)
                p.set_edgecolor(edgecolor)
            super().redraw()

    plot, ax = bar1d(data[0], axis, title, ylabel, filled, fill_scheme, edgecolor, grid, *args, **kwargs)
    plt.subplots_adjust(bottom=0.2)
    callback = BarPlotSlice(data, ax)
    ax_prev = plt.axes([0.7, 0.02, 0.1, 0.075])
    ax_next = plt.axes([0.81, 0.02, 0.1, 0.075])
    b_next = Button(ax_next, 'Next')
    b_next.on_clicked(callback.next)
    b_prev = Button(ax_prev, 'Previous')
    b_prev.on_clicked(callback.prev)
    return plot, ax, (b_next, b_prev)
