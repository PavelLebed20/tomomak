"""Routines to work with geometry
"""
import numpy as np
from shapely.geometry import Polygon, Point
import tomomak.util.array_routines


def intersection_2d(mesh, points, multuplier=1, index=(0, 1), norm_to=None, distant_point = None):
    """Create solution array, representing 2d polygon, defined by specified points on the given mesh.

    If there are more than 2 dimension in model, broadcasting to other dimensions is performed.
    If broadcasting is not needed private method _polygon may be used.
    Only axes which implements cell_edges2d method are supported.
    cell_edges2d method should  accept second axe and return 2d list of ordered sequence of point tuples for two 1d axes
    or 1d list of ordered sequence of point tuples for one 2d axis.
    Each point tuple represents cell borders in the 2D cartesian coordinates.
    E.g. borders of the cell of two cartesian axes with edges (0,7) and (0,5)
    is a rectangle which can be represented by the following point tuple ((0 ,0), (0, 7), (5,7), (5, 0)).
    Shapely module is used for the calculation.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        points(An ordered sequence of point tuples, optional): Polygon points (x, y).
            default: ((0 ,0), (5, 5), (10, 0))
        index(tuple of two ints, optional): axes to build object at. Default:  (0,1)
        multuplier(float, optional): Object density. Default: 1.
        broadcast(bool, optional) If true, resulted array is broadcasted to fit Mesh shape.
            If False, 2d array is returned, even if Mesh is not 2D. Default: True.

    Returns:
        ndarray: 2D numpy array, representing polygon on the given mesh.

    Raises:
        TypeError if one of the axes is not  cartesian (tomomak.main_structures.mesh.cartesian).
    """
    if norm_to is not None:
        if norm_to == 'distance':
            if distant_point is None:
                raise ValueError("distant_point should be defined.")
        if norm_to == 'cell_area':
            norm_to = 0
        elif norm_to == 'distance':
            norm_to = 1
        elif norm_to == 'distance2':
            norm_to = 2
        else:
            raise TypeError("divide_by type unknown. Possible types are 'cell_area', 'distance'.")

    if isinstance(index, int):
        index = [index]
    pol = Polygon(points)
    # If axis is 2D
    if mesh.axes[index[0]].dimension == 2:
        i1 = index[0]
        try:
            cells = mesh.axes[i1].cell_edges()
            shape = (mesh.axes[i1].size,)
            res = np.zeros(shape)
            for i, row in enumerate(res):
                cell = Polygon(cells[i])
                res[i] = pol.intersection(cell).area
                res[i] *= multuplier
                if norm_to is not None:
                    if norm_to == 0:
                        ds = cell.area
                    elif norm_to == 1:
                        ds = Point(distant_point).distance(Point(mesh.axes[i1].coordinates[i]))
                    elif norm_to == 2:
                        ds = Point(distant_point).distance(Point(mesh.axes[i1].coordinates[i]))**2
                    res[i] /= ds
            return res
        except (TypeError, AttributeError) as e:
            raise type(e)(e.message + "Custom axis should implement cell_edges method. "
                                      "This method returns 1d list of ordered sequence of point tuples."
                                      " See polygon method docstring for more information.")
    # If axes are 1D
    elif mesh.axes[0].dimension == 1:
        i1 = index[0]
        i2 = index[1]
        try:
            cells = mesh.axes[i1].cell_edges2d(mesh.axes[i2])
        except (TypeError, AttributeError):
            try:
                cells = mesh.axes[i2].cell_edges2d(mesh.axes[i1])
            except (TypeError, AttributeError) as e:
                raise type(e)(e.message + "Custom axis should implement cell_edges2d method. "
                                          "This method returns 2d list of ordered sequence of point tuples."
                                          " See polygon method docstring for more information.")
    else:
        raise TypeError("2D objects can be built on the 1D and 2D axes only.")
    shape = (mesh.axes[i1].size, mesh.axes[i2].size)
    res = np.zeros(shape)
    for i, row in enumerate(res):
        for j, _ in enumerate(row):
            cell = Polygon(cells[i][j])
            res[i, j] = pol.intersection(cell).area
            res[i, j] *= multuplier
            if norm_to is not None:
                if norm_to == 0:
                    ds = cell.area
                elif norm_to == 1:
                    ds = Point(distant_point).distance(Point(mesh.axes[i1].coordinates[i],
                                                        mesh.axes[i2].coordinates[j]))
                elif norm_to == 2:
                    ds = Point(distant_point).distance(Point(mesh.axes[i1].coordinates[i],
                                                             mesh.axes[i2].coordinates[j]))**2
                res[i, j] /= ds

    return res