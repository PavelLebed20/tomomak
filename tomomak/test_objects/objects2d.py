"""Functions for creation of different 2d objects.

Synthetic object are usually used to test different tomomak components.
"""
import numpy as np
from shapely.geometry import Polygon, Point
import shapely.affinity
from tomomak.main_structures.mesh.cartesian import Axis1d


def polygon(mesh, points=((0, 0), (5, 5), (10, 0)), density=1):
    """Create solution array, representing 2d polygon, defined by specified points on the given mesh.

    Only cartesian axes (tomomak.main_structures.mesh.cartesian) are supported.
    Shapely module is used for calculation.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        points(An ordered sequence of point tuples, optional): Polygon points (x, y).
            default: ([)(0 ,0), (5, 5), (10, 0))
        density(float, optional): Object density. default: 1.

    Returns:
        ndarray: 2D numpy array, representing polygon on the given mesh.

    Raises:
        TypeError if one of the axes is not  cartesian (tomomak.main_structures.mesh.cartesian).
    """
    if type(mesh.axes[0]) is not Axis1d or type(mesh.axes[1]) is not Axis1d:
        raise TypeError("Only cartesian axes (tomomak.main_structures.mesh.cartesian) are supported.")
    pol = Polygon(points)
    edge1 = mesh.axes[0].cell_edges
    edge2 = mesh.axes[1].cell_edges
    shape = (mesh.axes[0].size, mesh.axes[1].size)
    res = np.zeros(shape)
    for i, row in enumerate(res):
        for j, _ in enumerate(row):
            cell = Polygon([(edge1[i], edge2[j]), (edge1[i + 1], edge2[j]),
                            (edge1[i + 1], edge2[j + 1]), (edge1[i], edge2[j + 1])])
            res[i, j] = pol.intersection(cell).area
            ds = cell.area
            res[i, j] *= density / ds
    return res


def rectangle(mesh, center=(0, 0), size=(10, 10), density=1):
    """Create solution array, representing 2d rectangle, defined by specified parameters.

    Only cartesian axes (tomomak.main_structures.mesh.cartesian) are supported.
    Shapely module is used for calculation.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        center(tuple of int, optional): Center of the rectangle, given by tuples with 2 elements(x, y). default: (0, 0).
        size (tuple of int, optional): Length and height of the rectangle,
            given by tuples with 2 elements(length, height). default: (10, 10).
        density(float, optional): Object density. default: 1.

    Returns:
        ndarray: 2D numpy array, representing rectangle on the given mesh.

    Raises:
        TypeError if one of the axes is not  cartesian (tomomak.main_structures.mesh.cartesian).
    """
    points = [(center[0] - size[0] / 2, center[1] - size[1] / 2), (center[0] + size[0] / 2, center[1] - size[1] / 2),
              (center[0] + size[0] / 2, center[1] + size[1] / 2), (center[0] - size[0] / 2, center[1] + size[1] / 2)]
    return polygon(mesh, points, density)


def ellipse(mesh, center=(0, 0), ax_len=(5, 5), density=1, resolution=32):
    """Create solution array, representing 2d ellipse, defined by specified parameters.

    Only cartesian axes (tomomak.main_structures.mesh.cartesian) are supported.
    Shapely module is used for calculation.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        center(tuple of int, optional): Center of the ellipse, given by tuples with 2 elements(x, y). default: (0, 0).
        ax_len (tuple of int, optional): Half-width and Half-height of the ellipse,
            given by tuples with 2 elements (a, b). default: (5, 5).
        density(float, optional): Object density. default: 1.
        resolution(integer, optional): Relative number of points, approximating ellipse. default: 32.

    Returns:
        ndarray: 2D numpy array, representing ellipse on the given mesh.

    Raises:
        TypeError if one of the axes is not  cartesian (tomomak.main_structures.mesh.cartesian).
    """
    points = Point(0, 0).buffer(1, resolution)
    points = shapely.affinity.scale(points, ax_len[0], ax_len[1])
    points = shapely.affinity.translate(points, center[0], center[1])
    return polygon(mesh, points, density)


def pyramid(mesh, center=(0, 0), size=(10, 10), height=1):
    """Create solution array, representing  2d rectangle defined by specified parameters
        with density changing as height of the quadrangular pyramid .

    Only cartesian axes (tomomak.main_structures.mesh.cartesian) are supported.
    Shapely module is used for calculation.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        center(tuple of int): Center of the pyramid, given by tuples with 2 elements(x, y). default: (0, 0).
        size (tuple of int): Length and height of the pyramid, given by tuples with 2 elements(length, height).
            default: (10, 10).
        height(float, optional): Pyramid max height. Minimum height is 0. default: 1.

    Returns:
        ndarray: 2D numpy array, representing pyramid on the given mesh.

    Raises:
        TypeError if one of the axes is not  cartesian (tomomak.main_structures.mesh.cartesian).
    """
    rect = rectangle(mesh, center, size, height)
    mask = np.zeros(rect.shape)
    coord = [mesh.axes[0].coordinates, mesh.axes[1].coordinates]
    for i, row in enumerate(mask):
        for j, _ in enumerate(row):
            cell_coord = [coord[0][i], coord[1][j]]
            mask[i, j] = 1 - max(np.abs((cell_coord[0] - center[0]) / (size[0])),
                                 np.abs((cell_coord[1] - center[1]) / (size[1]))) * 2
    mask = mask.clip(min=0)
    return rect * mask


def cone(mesh, center=(3, 4), ax_len=(4, 3), height=1, resolution=32):
    """Create solution array, representing  2d ellipse defined by specified parameters
        with density changing as height of the elliptical cone .

    Only cartesian axes (tomomak.main_structures.mesh.cartesian) are supported.
    Shapely module is used for calculation.

    Args:
        mesh(tomomak.main_structures.Mesh): mesh to work with.
        center(tuple of int, optional): Center of the ellipse, given by tuples with 2 elements(x, y). default: (0, 0).
        ax_len (tuple of int, optional): Half-width and Half-height of the base ellipse,
            given by tuples with 2 elements (a, b). default: (5, 5).
        height(float, optional): Cone max height. Minimum height is 0. default: 1
        resolution(integer, optional): Relative number of points, approximating base ellipse. default: 32.

    Returns:
        ndarray: 2D numpy array, representing cone on the given mesh.

    Raises:
        TypeError if one of the axes is not  cartesian (tomomak.main_structures.mesh.cartesian).
    """
    ell = ellipse(mesh, center, ax_len, height, resolution)
    mask = np.zeros(ell.shape)
    coord = [mesh.axes[0].coordinates, mesh.axes[1].coordinates]
    for i, row in enumerate(mask):
        for j, _ in enumerate(row):
            cell_coord = [coord[0][i], coord[1][j]]
            mask[i, j] = 1 - np.sqrt(np.abs((cell_coord[0] - center[0]) / (ax_len[0]))**2 + np.abs(
                (cell_coord[1] - center[1]) / (ax_len[1]))**2)
    mask = mask.clip(min=0)
    return ell * mask
