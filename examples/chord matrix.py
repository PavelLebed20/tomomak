from math import acos, cos, sin, sqrt
import numpy as np
from numpy.linalg import norm

from tomomak.detectors.detector import Detector


def get_detectors():
    ang = acos((708 ** 2 + 720 ** 2 - 31 ** 2) / (2 * 708 * 720))
    spd_start = np.array([0, -0.708])
    spd_end = np.array([0.72 * sin(ang), 0.72 * -cos(ang)])
    spd_vect = (spd_end - spd_start) / norm(spd_end - spd_start)
    min_step = (2.3375 - 0.88) * 1e-03
    max_step = (3.81 - 2.3375 + 0.88) * 1e-03
    pp = spd_start + spd_vect * ((min_step + max_step) * 8 + 0.52 * 1e-03) / 2
    aperture_xy_offset = 0.0395
    aperture_xy = np.array([pp[0] - spd_vect[1] * aperture_xy_offset, pp[1] + spd_vect[0] * aperture_xy_offset])
    spd_z_start = (27.52 - 0.49) / 2 * 1e-03
    spd_z_step = -1.72 * 1e-03
    spd_xy = spd_start + spd_vect * (max_step / 2 + 0.26 * 1e-03)

    step = [[min_step, -min_step], [max_step, -max_step]]
    points_z = np.array([spd_z_start + i * spd_z_step for i in range(16)])
    points_xy = np.full((16, 2), spd_start + step[0])
    for j in range(1, 16):
        points_xy[j] = points_xy[j - 1] + spd_vect * (min_step if j % 2 == 1 else max_step)

    detectors = []
    for i in range(16):
        for j in range(16):
            origin = [*points_xy[i], points_z[j]]
            origin = np.array(origin)
            detectors.append(Detector(origin, aperture_xy, spd_xy))

    return np.array(detectors)


if __name__ == "__main__":
    pass
