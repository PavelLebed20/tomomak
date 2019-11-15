from tomomak.main_structures.Mesh.mesh import *
from tomomak.main_structures.model import *
from tomomak.main_structures.Mesh.cartesian import *


#axes = [Axis1d(name="x", units="cm", size=2), Axis1d(name="Y", units="cm", size=3)]
axes = [Axis1d(name="x", units="cm", size=2), Axis1d(name="Y", units="cm", coordinates=np.array([55, 66, 99]), lower_limit=52)]
mesh = Mesh(axes)
detector_geometry = np.array([[[0, 1, 4], [1, 2,4 ]], [[0, 1, 3 ], [1, 5,3 ]], [[0, 1,3], [3, 2,3]], [[0,2, 3 ], [1, 1,22 ]]])
detector_signal=np.array([3, 1, 5, 4])
solution = np.array([[1, 2, 1], [5, 5.1, 4]])

mod = Model( detector_geometry, detector_signal, solution, mesh)
mod.plot2d()
#mod.plot1d(index=1, data_type='detector_geometry')
#mod.plot1d(index=1,data_type='detector_geometry', filled=False)
print(mod )