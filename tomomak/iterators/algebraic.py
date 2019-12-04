from . import abstract_iterator
import numpy as np
import warnings
from tomomak.detectors import signal

class ART(abstract_iterator.AbstractIterator):
    """A set of iterative algebraic algorithms for image reconstruction
    see E.F. Oliveira et. al., Comparison among tomographic reconstruction algorithms with limited data
    in the case of ART correction is applied after calculations of single ray
    in case of SIRT averaged correction is applied at the end of iteration.
    In the case of several projections in SIRT averaged correction is applied after iteration over each slide
    """

    def __init__(self, alpha):
        self.shape = None
        self.wi = None
        self.alpha = alpha

    def init(self, model):  # maybe make this __init__
        if model.solution is None:
            shape = model.mesh.shape
            model.solution = np.zeros(shape)
        self.shape = model.solution.shape
        self.wi = np.sum(np.square(model.detector_geometry), axis=tuple(range(1, model.detector_geometry.ndim)))

    def finalize(self, model):
        pass

    @property
    def name(self):
        return 'Maximum Likelihood method'

    def step(self, model):
        # expected signal
        expected_signal = signal.get_signal(model.solution, model.detector_geometry)
        # multiplication
        new_sol  = model.solution
        for i in range(model.detector_signal.shape[0]):
            y =  expected_signal[i]
            dp = model.detector_signal[i] - y
            if (self.wi[i] != 0):
                ai = dp / self.wi[i]
            else:
                ai = 0
            # if type == 'MART':
            #     if model.detector_signal[i] != 0:
            #         ai = ai / np.abs(model.detector_signal.y[i])
            # res = res + param * ai * w[i]
            new_sol =  new_sol  + ai * model.detector_geometry[i] * self.alpha
        delta = new_sol - model.solution
        return delta
