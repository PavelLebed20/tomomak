import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.widgets import Button


def colormesh2d(data, axis1, axis2, title='', fill_scheme='viridis', grid=False, *args,  **kwargs):
    """

    """
    x = axis1.cell_edges
    y = axis2.cell_edges
    z = data.transpose()
    cmap = plt.get_cmap(fill_scheme)
    fig, ax = plt.subplots()
    plot = ax.pcolormesh(x, y, z, cmap=cmap, *args,  **kwargs)
    fig.colorbar(plot, ax=ax)
    ax.set_title(title, loc='right')
    xlabel = "{}, {}".format(axis1.name, axis1.units)
    ylabel = "{}, {}".format(axis2.name, axis2.units)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if grid:
        ax.grid()
    return plot, ax