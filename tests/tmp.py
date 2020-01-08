from tomomak.model import *
from tomomak.solver import *
from tomomak.test_objects.objects2d import *
from tomomak.mesh.mesh import *
from tomomak.mesh.cartesian import Axis1d
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak import iterators
from tomomak.iterators import ml, algebraic#, gpu
from tomomak.iterators import statistics
import tomomak.constraints.basic
from mpl_toolkits.mplot3d import Axes3D
import itertools


axes = [Axis1d(name="X", units="cm", size=15, upper_limit=10),
        Axis1d(name="Y", units="cm", size=15, upper_limit=10)]
mesh = Mesh(axes)
# Now we can create Model.
# Model is one of the basic tomomak structures which stores information about geometry, solution and detectors.
# At present we only have information about the geometry.
mod = Model(mesh=mesh)
# Now let's create synthetic 2D object to study.
# We will consider triangle.
real_solution = polygon(mesh, [(1, 1), (4, 8), (7, 2)])
# Model.solution is the solution we are looking for.
# It will be obtained at the end of this example.
# However, if you already know supposed solution (for example you get it with simulation),
# you can use it as first approximation by setting Model.solution = *supposed solution*.
# Recently we've generated test object, which is, of course, real solution.
# A trick to visualize this object is to temporarily use it as model solution.
mod.solution = real_solution

# After we've visualized our test object, it's time to set model solution to None and try to find this solution fairly.
mod.solution = None

# Next step is to provide information about the detectors.
# Let's create 15 fans with 22 detectors around the investigated object.
# Each line will have 1 cm width and 0.2 Rad divergence.
# Note that number of detectors = 330 < solution cells = 600, so it's impossible to get perfect solution.
det = detectors2d.fan_detector_array(mesh=mesh,
                                     focus_point=(5, 5),
                                     radius=11,
                                     fan_num=1,
                                     line_num=22,
                                     width=1,
                                     divergence=0.2)
# Now we can calculate signal of each detector.
# Of course in the real experiment you measure detector signals so you don't need this function.
det_signal = signal.get_signal(real_solution, det)
mod.detector_signal = det_signal
mod.detector_geometry = det
# Let's take a look at the detectors geometry:
mod.plot2d(data_type='detector_geometry', equal_norm=True)
mod.plot1d(data_type='detector_geometry',equal_norm=True )
#axes = [Axis1d(name="x", units="cm", size=20), Axis1d(name="Y", units="cm", size=30), Axis1d(name="Y", units="cm", size=130)]
axes = [Axis1d(name="x", units="cm", size=5), Axis1d(name="Y", units="cm", size=10), Axis1d(name="Z", units="cm", size=20)]
#axes = [Axis1d(name="x", units="cm", size=21), Axis1d(name="Y", units="cm", coordinates=np.array([1, 3, 5, 7, 9, 13]),  lower_limit=0), Axis1d(name="z", units="cm", size=3)]


# inters = axes[0].cell_edges3d(axes[1], axes[2])
# rect = inters[0][0][0]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
#
# for p in rect:
#     xs = p[0]
#     ys = p[1]
#     zs = p[2]
#     ax.scatter(xs, ys, zs)
# plt.show()

mesh = Mesh(axes)

solution = polygon(mesh, [(1,1), (1, 8), (7, 9), (7, 2)])

#solution = detectors2d.line2d(mesh, (-1, 7), (11, 3), 1, divergence=0.1, )
det = detectors2d.fan_detector_array(mesh, (5,5), 11, 10, 22, 1, incline=0 )

det_signal = signal.get_signal(solution, det)
#det = detectors2d.parallel_detector(mesh,(-10, 7), (11, 3), 1, 10, 0.2)

#det = detectors2d.fan_detector(mesh, (-3, 7), (11, 7), 0.5, 10, angle=np.pi/2)
# solution = rectangle(mesh,center=(6, 4), size = (4, 2.7), index = (1,2))
# solution  = real_solution(mesh)
# solution  = pyramid(mesh,center=(6, 4), size = (6.1, 2.7) )
# solution  = cone(mesh,center=(5, 5), ax_len=(3, 7))

# detector_geometry = np.array([[[0, 1, 4], [1, 2,4 ]], [[0, 1, 3 ], [1, 5,3 ]], [[0, 8,3], [3, 2,3]], [[0,2, 3 ], [1, 1, 22 ]]])
#detector_geometry = np.array([[[0, 0, 0], [0, 0,0 ]], [[0, 0, 0 ], [0, 0,0 ]], [[0, 0,0], [0, 0,0]], [[0,0, 0 ], [0, 0, 0 ]]])
# detector_signal=np.array([3, 1, 5, 4])
# solution = np.array([[1, 2, 1], [5, 5.1, 4]])
#solution= sparse.COO(solution)

mod = Model(mesh=mesh,  detector_signal = det_signal, detector_geometry=det, solution = solution)
mod.plot3d()
#mod.plot2d(index=(0,1))
mod.solution = None
solver = Solver()
steps = 100
solver.real_solution = solution
import cupy as cp
solver.iterator = ml.ML()
# solver.alpha = cp.linspace(1, 1, steps)
#solver.iterator = gpu.MLCuda()
#solver.iterator.alpha = cp.linspace(1, 1, steps)
solver.statistics = [statistics.rms]
# solver.alpha = np.linspace(1, 1, steps)
#solver.iterator = algebraic.ART()
#solver.iterator = algebraic.SIRT(n_slices=3, iter_type='SIRT')
solver.iterator.alpha =  np.linspace(0.1, 0.0001, steps)

import scipy.ndimage
func = scipy.ndimage.gaussian_filter1d
#c2 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=0, alpha=1, sigma=2)
c2 = tomomak.constraints.basic.ApplyFunction(scipy.ndimage.gaussian_filter, sigma=1, alpha=1)
# c3 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=1, alpha=1, sigma=2)
solver.constraints = [tomomak.constraints.basic.Positive(), c2]
import time
start_time = time.time()
solver.stop_conditions = [statistics.rms]
solver.stop_values = [0.2]
solver.solve(mod, steps = steps)
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