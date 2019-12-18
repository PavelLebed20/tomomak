from tomomak.model import *
from tomomak.solver import *
from tomomak.test_objects.objects2d import ellipse
from tomomak.mesh.mesh import *
from tomomak.mesh.cartesian import Axis1d
from tomomak.transform import rescale
from tomomak.transform import pipeline
from tomomak.detectors import detectors2d, signal
from tomomak.iterators import ml, algebraic
from tomomak.iterators import statistics
import tomomak.constraints.basic
import matplotlib.pyplot as plt
import scipy.ndimage


axes = [Axis1d(name="X", units="cm", size=20),
        Axis1d(name="Y", units="cm", size=30)]
mesh = Mesh(axes)
solution = ellipse(mesh, center=(5, 5), ax_len=(3, 3),  density=1, resolution=32)
mod = Model(mesh=mesh, solution=solution)
# mod.plot2d()
# mod.plot1d(index=0)

f1d = mesh.integrate_other(solution, 0)
np.savetxt("f1d_real.txt", f1d)
mod.solution = None

det = detectors2d.fan_detector_array(mesh=mesh,
                                     focus_point=(5, 5),
                                     radius=11,
                                     fan_num=30,
                                     line_num=40,
                                     width=0.5,
                                     divergence=0.1,
                                     response=1000)

det_signal = signal.get_signal(solution, det)
det_signal = signal.add_noise(det_signal, 10)
mod.detector_signal = det_signal
mod.detector_geometry = det
print(mod)
#mod.plot2d(data_type='detector_geometry')


solver = Solver()
steps =1000
solver.real_solution = solution
solver.stat_array = [statistics.rms, statistics.rn, statistics.chi_sc]
solver.iterator = algebraic.ART()
#solver.iterator = ml.ML()
solver.iterator.alpha = 0.1
func = scipy.ndimage.gaussian_filter1d
res = ""
dat = []
# for alpha in np.linspace(0.01, 0.32, 30):
#     for sig in np.linspace(0.2, 2, 20):
#         mod.solution = None
#         c1 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=0, alpha=alpha, sigma=sig)
#         c2 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=1, alpha=alpha, sigma=sig)
#         solver.constraints_array = [tomomak.constraints.basic.Positive(), c1, c2]
#         solver.solve(mod, steps=steps)
#         res+= "alpha = {}, sigma = {}, rms = {}\n".format(alpha, sig, solver.statistics[-1][0])
#         new_dat = [alpha, sig, solver.statistics[-1][0]]
#         dat.append(new_dat)
# print(res)
# np.savetxt("grid_search_stat.txt", np.array(dat))
# mod.plot2d()
#
# solver.solve(mod, steps=steps)
# mod.plot2d()
# f1d = mesh.integrate_other(mod.solution, 0)

# np.savetxt("f1d_reconstr50.txt", f1d)
#
# steps = 100
# solver.solve(mod, steps=steps)
# mod.plot2d()
#
# steps = 800
# solver.solve(mod, steps=steps)
# #mod.plot2d()
# print(np.array(solver.statistics))
# np.savetxt("f1d_stat.txt", np.array(solver.statistics))
#
# plt.plot(solver.statistics)
#plt.show()

mod.solution = None
alpha = 0.18
sig = 0.75
c1 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=0, alpha=alpha, sigma=sig)
c2 = tomomak.constraints.basic.ApplyAlongAxis(func, axis=1, alpha=alpha, sigma=sig)
solver.constraints_array = [tomomak.constraints.basic.Positive(), c1, c2]
solver.solve(mod, steps=steps)
np.savetxt("f1d_stat.txt", np.array(solver.statistics))
f1d = mesh.integrate_other(mod.solution, 0)
np.savetxt("f1d_10000.txt", f1d)
mod.plot2d()
# pipe = pipeline.Pipeline(mod)
# r = rescale.Rescale((200, 30))
# pipe.add_transform(r)
# pipe.forward()
# mod.plot2d()
# pipe.backward()
#C

