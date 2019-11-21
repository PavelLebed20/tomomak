from tomomak.model import *
from tomomak.test_objects.objects2d import *
from tomomak.mesh.mesh import *
from tomomak.mesh.cartesian import Axis1d

a = np.arange(6).reshape(2,3)
with np.nditer(a, op_flags=['readwrite']) as it:
    for x in it:
        print(x[...])


x = np.array([1, 2, 3])
b = np.broadcast_to(x, (2, 3))
c = np.broadcast_to(b, (4,2,3))
print(c.shape)
print(np.swapaxes(c, 0,1).shape)

axes = [Axis1d(name="x", units="cm", size=150), Axis1d(name="Y", units="cm", size=140), Axis1d(name="Y", units="cm", size=130)]
#axes = [Axis1d(name="x", units="cm", size=2), Axis1d(name="Y", units="cm", coordinates=np.array([55, 66, 99]), lower_limit=52)]

mesh = Mesh(axes)
solution = polygon(mesh, [(1,1), (1, 8), (7, 8), (7, 2)])
solution = rectangle(mesh,center=(6, 4), size = (4, 2.7), index = (1,2))
solution  = ellipse(mesh)
solution  = pyramid(mesh,center=(6, 4), size = (6.1, 2.7) )
solution  = cone(mesh,center=(5, 5), ax_len=(3, 7))


# detector_geometry = np.array([[[0, 1, 4], [1, 2,4 ]], [[0, 1, 3 ], [1, 5,3 ]], [[0, 8,3], [3, 2,3]], [[0,2, 3 ], [1, 1, 22 ]]])
#detector_geometry = np.array([[[0, 0, 0], [0, 0,0 ]], [[0, 0, 0 ], [0, 0,0 ]], [[0, 0,0], [0, 0,0]], [[0,0, 0 ], [0, 0, 0 ]]])
# detector_signal=np.array([3, 1, 5, 4])
# solution = np.array([[1, 2, 1], [5, 5.1, 4]])
#solution= sparse.COO(solution)

mod = Model(solution=solution, mesh=mesh)
mod.plot2d(index=(0,1))
#mod.plot1d(index=1, data_type='detector_geometry')
mod.plot1d(index=1)
print(mod)