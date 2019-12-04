from tomomak.model import *
from tomomak.solver import *
from tomomak.test_objects.objects2d import *
from tomomak.mesh.mesh import *
from tomomak.mesh.cartesian import Axis1d
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak import iterators
from tomomak.iterators import ml, algebraic
from tomomak.iterators import statistics
import tomomak.regularization.basic



axes = [Axis1d(name="x", units="cm", size=20), Axis1d(name="Y", units="cm", size=30), Axis1d(name="Y", units="cm", size=130)]
#axes = [Axis1d(name="x", units="cm", size=21), Axis1d(name="Y", units="cm", coordinates=np.array([1, 3, 5, 7, 9, 13]),  lower_limit=0), Axis1d(name="z", units="cm", size=3)]

mesh = Mesh(axes)

solution = polygon(mesh, [(1,1), (1, 8), (7, 9), (7, 2)])

#solution = detectors2d.line2d(mesh, (-1, 7), (11, 3), 1, divergence=0.1, )
det = detectors2d.fan_detector_array(mesh, (5,5), 11, 20, 22, 1, incline=0 )

det_signal = signal.get_signal(solution, det)
#det = detectors2d.parallel_detector(mesh,(-10, 7), (11, 3), 1, 10, 0.2)

#det = detectors2d.fan_detector(mesh, (-3, 7), (11, 7), 0.5, 10, angle=np.pi/2)
# solution = rectangle(mesh,center=(6, 4), size = (4, 2.7), index = (1,2))
# solution  = ellipse(mesh)
# solution  = pyramid(mesh,center=(6, 4), size = (6.1, 2.7) )
# solution  = cone(mesh,center=(5, 5), ax_len=(3, 7))

# detector_geometry = np.array([[[0, 1, 4], [1, 2,4 ]], [[0, 1, 3 ], [1, 5,3 ]], [[0, 8,3], [3, 2,3]], [[0,2, 3 ], [1, 1, 22 ]]])
#detector_geometry = np.array([[[0, 0, 0], [0, 0,0 ]], [[0, 0, 0 ], [0, 0,0 ]], [[0, 0,0], [0, 0,0]], [[0,0, 0 ], [0, 0, 0 ]]])
# detector_signal=np.array([3, 1, 5, 4])
# solution = np.array([[1, 2, 1], [5, 5.1, 4]])
#solution= sparse.COO(solution)

mod = Model(mesh=mesh,  detector_signal = det_signal, detector_geometry=det, solution = solution)
#mod.plot2d(index=(0,1))
mod.solution = None
solver = Solver()
steps = 500

import cupy as cp
# solver.alpha = np.linspace(1, 1, steps)
# solver.iterator = ml.ML()
# solver.alpha = cp.linspace(1, 1, steps)
# solver.iterator = ml.MLCuda()
#solver.stat_array = [statistics.rms]
solver.alpha = np.linspace(1, 1, steps)
solver.iterator = algebraic.ART(0.1)
solver.reg_alpha = [1]
solver.reg_array = [tomomak.regularization.basic.positive]
import time
start_time = time.time()
solver.solve(mod, steps = steps, real_solution=solution)
print("--- %s seconds ---" % (time.time() - start_time))
mod.plot2d(index=(0,1), data_type='detector_geometry')
#mod.plot1d(index=0, data_type='detector_geometry')

# pipe = pipeline.Pipeline(mod)
# r = rescale.Rescale((80, 80, 80))
# pipe.add_transform(r)
# pipe.forward()
mod.plot2d(index=(0,1))
# pipe.backward()
# mod.plot2d(index=(0,1))
# #mod.plot1d(index=1, data_type='detector_geometry')
# mod.plot1d(index=1)
print(mod)
# print(len(solver.statistics))