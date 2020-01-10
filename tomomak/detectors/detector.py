import numpy as np

from math import sqrt

from tomomak.util.geometry2d import create_line, get_line_value


class Detector(object):
    def __init__(self, origin, aperture_xy, spd_xy):
        """Constructor of detector

        Args:
            origin(ndarray): detector's position
            aperture_xy(ndarray): aperture's coordinates in XY system
            spd_xy(ndarray): center of detectors' columns line
        """

        self.origin = origin
        self.line_xy = create_line(np.array([origin[0], origin[1]]), aperture_xy)

        # self.line_xy[2][2] = c, where `ax + by = c` is self.line_xy
        spd_r = -sqrt(np.linalg.norm(spd_xy) ** 2 - self.line_xy[2] ** 2)
        aperture_xz_offset = np.linalg.norm(spd_xy - aperture_xy)
        line_xz = create_line([spd_r, self.origin[2]], [aperture_xz_offset + spd_r, 0])

        self.line_xz_points = [[spd_r, self.origin[2]], [0.6, get_line_value(0.6, line_xz)]]
