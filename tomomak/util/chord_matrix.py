from shapely.geometry import Polygon, LineString

import numpy as np

from tomomak.util.domains import get_domains
from tomomak.util.section import sorted_section_by_nonzero_plane


def _get_intersection_length(points, line_points):
    """Function for getting length of intersection of polygon and line

    Args:
        points(ndarray): points that represent the polygon
        line_points(ndarray): points that represent the line

    Returns:
        float: length of the intersection
    """

    length = 0
    polygon = Polygon(points)
    line = LineString(line_points)
    intersection = polygon.intersection(line)
    if not intersection.is_empty:
        if intersection.geom_type == 'MultiLineString':
            for line in intersection.geoms:
                length += line.length
        else:
            length += intersection.length
    return length


def generate_chord_matrix(border, center, detectors, count=0, radials=0):
    """Function for generation matrix of chords' lengths

    Args:
        border(ndarray): outer border line that represents by ndarrays; closed curve, but border[0] is not equal to border[-1]
        center(ndarray): two digits - point of the center of the grid
        detectors(ndarray(Detector)): array of needed detectors for creating chord matrix
        count(int): count of lines to generate; must be non-negative
        radials(int): count of radial lines between four support radial lines; must be non-negative

    Returns:
        ndarray: generated matrix of chords' lengths
    """

    domains = get_domains(border, center, count=count, radials=radials)

    matrix = np.zeros((256, (count + 1) * (radials + 1) * 4))

    for j in range(16):
        # every 16 elements in detectors array is a column of detectors
        col_of_detectors = detectors[j * 16: (j + 1) * 16]
        col_line_xy = col_of_detectors[0].line_xy
        line_points = [det.line_xz_point for det in col_of_detectors]
        for i, domain in enumerate(domains[:]):
            section = sorted_section_by_nonzero_plane(domain, np.array([1, 0, col_line_xy[2]]))
            if section.shape[0] > 0:
                if section.shape[0] >= 2 * domain.shape[0]:
                    left = section[section[:, 1] <= 0, 1:]
                    right = section[section[:, 1] >= 0, 1:]
                    for k, l_p in enumerate(line_points):
                        matrix[j * 16 + k, i] += _get_intersection_length(left, l_p)
                    if j < 12:
                        for k, l_p in enumerate(line_points):
                            matrix[j * 16 + k, i] += _get_intersection_length(right, l_p)
                elif section.shape[0] > 2:
                    points = section[:, 1:]
                    for k, l_p in enumerate(line_points):
                        matrix[j * 16 + k, i] += _get_intersection_length(points, l_p)
    return matrix
