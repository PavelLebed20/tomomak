from tomomak.model import *
from tomomak.solver import *
from tomomak.test_objects.objects2d import *
from tomomak.mesh.mesh import *
from tomomak.mesh.cartesian import Axis1d
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak.iterators import ml, algebraic
from tomomak.iterators import statistics
import tomomak.constraints.basic
import matplotlib.pyplot as plt


axes = [Axis1d(name="X", units="cm", size=20),
        Axis1d(name="Y", units="cm", size=30)]
mesh = Mesh(axes)
solution = polygon(mesh, [(1, 1), (4, 8), (7, 2)])
mod = Model(mesh=mesh, solution=solution)
mod.plot2d()
mod.plot1d(index=0)
mod.solution = None

det = detectors2d.fan_detector_array(mesh=mesh,
                                     focus_point=(5, 5),
                                     radius=11,
                                     fan_num=15,
                                     line_num=22,
                                     width=1,
                                     divergence=0.2)

det_signal = signal.get_signal(solution, det)
mod.detector_signal = det_signal
mod.detector_geometry = det
print(mod)
mod.plot2d(data_type='detector_geometry')


solver = Solver()
steps = 500
solver.real_solution = solution
solver.stat_array = [statistics.rms]
solver.iterator = algebraic.ART()
solver.iterator.alpha = 0.1
solver.constraints_array = [tomomak.constraints.basic.Positive()]
solver.solve(mod, steps=steps)
mod.plot2d()

solver.iterator = ml.ML()
solver.stop_array = [statistics.rms]
solver.stop_values = [0.07]
steps = 1000
solver.solve(mod, steps=steps)
mod.plot2d()

plt.plot(solver.statistics)
plt.show()


pipe = pipeline.Pipeline(mod)
r = rescale.Rescale((200, 30))
pipe.add_transform(r)
pipe.forward()
mod.plot2d()
pipe.backward()
mod.plot2d()

