from tomomak.main_structures.mesh.mesh import *
from tomomak.main_structures.model import *
from tomomak.main_structures.mesh.cartesian import *
from tomomak.test_objects.objects2d import *
import sparse


axes = [Axis1d(name="x", units="cm", size=150), Axis1d(name="Y", units="cm", size=140)]
#axes = [Axis1d(name="x", units="cm", size=2), Axis1d(name="Y", units="cm", coordinates=np.array([55, 66, 99]), lower_limit=52)]

mesh = Mesh(axes)
#solution = polygon(mesh, [(1,1), (1, 8), (7, 8), (7, 2)])
# solution = rectangle(mesh,center=(6, 4), size = (3.1, 2.7))
# solution  = ellipse(mesh)
# solution  = pyramid(mesh,center=(6, 4), size = (6.1, 2.7) )
solution  = cone(mesh,center=(6, 4), ax_len=(3.1, 2.7))
# detector_geometry = np.array([[[0, 1, 4], [1, 2,4 ]], [[0, 1, 3 ], [1, 5,3 ]], [[0, 8,3], [3, 2,3]], [[0,2, 3 ], [1, 1, 22 ]]])
#detector_geometry = np.array([[[0, 0, 0], [0, 0,0 ]], [[0, 0, 0 ], [0, 0,0 ]], [[0, 0,0], [0, 0,0]], [[0,0, 0 ], [0, 0, 0 ]]])
# detector_signal=np.array([3, 1, 5, 4])
# solution = np.array([[1, 2, 1], [5, 5.1, 4]])
#solution= sparse.COO(solution)

mod = Model(solution=solution, mesh=mesh)
mod.plot2d()
#mod.plot1d(index=1, data_type='detector_geometry')
mod.plot1d(index=1)
print(mod )