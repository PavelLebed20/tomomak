from cmath import pi

import numpy as np

from tomomak.detectors.detectors import line_intersect
from tomomak.mesh import Axis1d, spiderweb_axes
from tomomak.util.gfileextractor import gfile_extract
from tomomak.util.geometry.geometry3d import  Geometry3d
from tomomak.mesh import mesh

#line = np.array([np.array([0, 0, -1]), np.array([0, 0, 1])])
#tri = np.array([np.array([-2, 2, 0]), np.array([4, -2, 0]), np.array([-2, -2, 0])])

#p = Geometry3d.triangle_line_intersection(tri, line)

border, center = gfile_extract('data/g035685.00150')
axes = [spiderweb_axes.SpiderWeb2dAxis(border=border, center=center), Axis1d(name="Y", units="cm", size=10, lower_limit=0, upper_limit=2 * pi)]
mesh = mesh.Mesh(axes)
intersect = line_intersect(mesh, (0, 0, 0), (5, 5, 0), 0.5, index=(0, 1), geometry=Geometry3d)
print(intersect)
