from math import sqrt

import numpy as np


def _get_intersection(r, z, plane):
    """Get points of intersection of a circle and a line

    Args:
        r(float): radius
        z(float): z
        plane(array_like): three digits - a, b and c for plane `ax + by = c`

    Returns:
        ndarray: points which represent the intersection of a circle and a line
    """

    a, b, c = plane
    s_len = a * a + b * b
    x0 = a * c / s_len
    y0 = b * c / s_len
    d = r * r - c * c / s_len
    if d < -1e-9:
        return []
    elif abs(d) < 1e-9:
        return np.array([(x0, y0, z)])
    else:
        delta = sqrt(d / s_len)
        x1 = x0 + b * delta
        x2 = x0 - b * delta
        y1 = y0 - a * delta
        y2 = y0 + a * delta
        return np.array([(x1, y1, z), (x2, y2, z)])


def sorted_section_by_nonzero_plane(obj, plane):
    """Function for find section of some rotation figure by plane
       Work only with nonzero plane without tangents in section

    Args:
        obj(ndarray): points in format (radius, z) which represent the rotation figure
        plane(ndarray): three digits - a, b and c for plane `ax + by = c`; only one of a, b, c can be zero

    Returns:
        ndarray: ordered points which represent the section
    """

    res = [[], []]
    ind = -1
    for i, p in enumerate(obj):
        intersect = _get_intersection(p[0], p[1], plane)
        if len(intersect) == 0:
            ind = len(res[0])
        else:
            res[0].append(intersect[0])
            res[1].append(intersect[1])
    if ind != -1:
        res[0] = res[0][ind:] + res[0][:ind]
        res[1] = res[1][ind:] + res[1][:ind]
    res[1].reverse()
    return np.array(res[0] + res[1])
