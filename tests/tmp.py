from tomomak.model import *
from tomomak.test_objects.objects2d import *
from tomomak.mesh.mesh import *
from tomomak.mesh.cartesian import Axis1d
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d

a = np.arange(6).reshape(2,3)
with np.nditer(a, op_flags=['readwrite']) as it:
    for x in it:
        print(x[...])
x = np.array([[[1,11],[2, 22],[3, 33]], [[4, 44],[5, 44],[6, 66]]])
print(x[...,0].shape)
print(x[...,1])

axes = [Axis1d(name="x", units="cm", size=150), Axis1d(name="Y", units="cm", size=140), Axis1d(name="Y", units="cm", size=130)]
#axes = [Axis1d(name="x", units="cm", size=21), Axis1d(name="Y", units="cm", coordinates=np.array([1, 3, 5, 7, 9, 13]),  lower_limit=0), Axis1d(name="z", units="cm", size=3)]

mesh = Mesh(axes)
import time
start = time.time()
solution = polygon(mesh, [(1,1), (1, 8), (7, 9), (7, 2)])

solution = detectors2d.line2d(mesh, (-1, 7), (11, 3), 1, divergence=0.1, )
print(time.time() - start)
# solution = rectangle(mesh,center=(6, 4), size = (4, 2.7), index = (1,2))
# solution  = ellipse(mesh)
# solution  = pyramid(mesh,center=(6, 4), size = (6.1, 2.7) )
# solution  = cone(mesh,center=(5, 5), ax_len=(3, 7))


# detector_geometry = np.array([[[0, 1, 4], [1, 2,4 ]], [[0, 1, 3 ], [1, 5,3 ]], [[0, 8,3], [3, 2,3]], [[0,2, 3 ], [1, 1, 22 ]]])
#detector_geometry = np.array([[[0, 0, 0], [0, 0,0 ]], [[0, 0, 0 ], [0, 0,0 ]], [[0, 0,0], [0, 0,0]], [[0,0, 0 ], [0, 0, 0 ]]])
# detector_signal=np.array([3, 1, 5, 4])
# solution = np.array([[1, 2, 1], [5, 5.1, 4]])
#solution= sparse.COO(solution)

mod = Model(solution=solution, mesh=mesh)
mod.plot2d(index=(0,1))
# pipe = pipeline.Pipeline(mod)
# r = rescale.Rescale((80, 80, 80))
# pipe.add_transform(r)
# pipe.forward()
# mod.plot2d(index=(0,1))
# pipe.backward()
# mod.plot2d(index=(0,1))
# #mod.plot1d(index=1, data_type='detector_geometry')
# mod.plot1d(index=1)
print(mod)