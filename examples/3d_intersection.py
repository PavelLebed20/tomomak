from tomomak.detectors.detectors import line_intersect
from tomomak.mesh import Axis1d
from tomomak.util.gfileextractor import gfile_extract
from tomomak.util.geometry.geometry3d import  Geometry3d
from tomomak.mesh import mesh

border, center = gfile_extract('data/g035685.00150')
axes = [Axis1d(name="X", units="cm", size=30, lower_limit=0, upper_limit=10),
        Axis1d(name="Y", units="cm", size=30, lower_limit=0, upper_limit=10),
        Axis1d(name="Z", units="cm", size=30, lower_limit=0, upper_limit=10)]
mesh = mesh.Mesh(axes)
intersect = line_intersect(mesh, (0, 0, 0), (5, 5, 0), 0.5, index=(0, 1, 2), geometry=Geometry3d)
print(intersect)
