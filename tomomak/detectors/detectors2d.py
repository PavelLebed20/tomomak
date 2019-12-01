"""Generators for basic detectors arrays in 2D geometry
"""
from shapely.geometry import Polygon
from shapely.affinity import rotate
from shapely.geometry import LineString
import tomomak.util.geometry.intersection_2d
import numpy as np

def line2d(mesh, p1, p2, index=(0, 1), response=1, broadcast=True):
    """

    Args:
        mesh:
        p1(tuple of 2 floats): Detector origin (x, y).
        p2:
        index:
        response: ratio of signal to number of emitted particles at sphere with R = 1m
        broadcast:

    Returns:

    """
    points = _line_to_polygon(p1, p2)
    res = tomomak.util.geometry.intersection_2d(mesh, points, response, (0, 1), norm_to='distance2', distant_point=p1)
    if broadcast:
        res = tomomak.util.array_routines.broadcast_object(res, index, mesh.shape)



    return res

def _line_to_polygon(p1, p2, width, divergence=0):
    """Generate detector geometry for one Line of Sight.

    line of sight can be collimated or diverging (cone-like).

    Args:
        p1, p2(tuple of 2 points):
        width(float): line width.
        divergence(float, optional): Angle between two LOS borders in Rad [0, pi). 0 means that line is collimated.
            default: 0.

    Returns:
        tuple of 4 points: List of 4 Shapely points, defining LOS.

    Raises:
        ValueError if devirgence is < 0 or >= pi.
    """
    if divergence < 0 or divergence > np.pi:
        raise ValueError("Divergence value is {}. It should be >= 0 and < pi.".format(divergence))
    line = LineString([p1, p2])
    lr = line.parallel_offset(width/2, 'right')
    p_top = lr.coords[0]
    lr = rotate(lr, divergence/2, origin=p_top, use_radians=True)
    ll = line.parallel_offset(width/2, 'left')
    p_top = ll.coords[1]
    ll = rotate(ll, -divergence/2, origin=p_top, use_radians=True)
    p1, p2 = lr.coords
    p3, p4 = ll.coords
    return (p1, p2, p3, p4)






