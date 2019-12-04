from . import abstract_iterator
import numpy as np
import warnings
import cupy as cp


class MLCuda(abstract_iterator.AbstractIterator):

    def __init__(self):
        self.w_det = None
        self.wi = None
        self.shape = None
        self.y_expected = None
        self.y_len = None
        self.mult = None

    def init(self, model): #maybe make this __init__
        if model.solution is None:
            shape = model.mesh.shape
            model.solution = cp.ones(shape)
        else:
            if cp.all(model.solution):
                warnings.warn("Some elements in model solution are zero. They will not be changed.")
        warnings.warn("Cuda ML is an experimental feature.")
        self.shape = model.solution.shape
        self.w_det = cp.array(np.multiply(np.moveaxis(model.detector_geometry, 0, -1), model.detector_signal))
        self.wi = cp.array(np.sum(model.detector_geometry, axis=0))
        model.detector_geometry = cp.array(model.detector_geometry)
        model.solution = cp.array(model.solution)
        self.y_expected = cp.zeros(model.detector_signal.shape)
        self.y_len = self.y_expected.shape[0]
        self.mult = cp.array(self.w_det)


    def finalize(self, model):
        model.detector_geometry = cp.asnumpy(model.detector_geometry)
        model.solution = cp.asnumpy(model.solution)

    @property
    def name(self):
        return 'Maximum Likelihood method'

    def step(self, model):
        #expected signal
        for i in range(self.y_len):
            tmp = cp.multiply(model.solution, model.detector_geometry[i])
            self.y_expected[i] = cp.sum(tmp)
        # multiplication
        self.mult = cp.divide(self.w_det, self.y_expected)
        self.mult = cp.where(cp.isnan(self.mult), 0, self.mult)
        self.mult = cp.sum(self.mult, axis=-1)
        self.mult /= self.wi
        # find delta
        new_sol = model.solution * self.mult
        delta = new_sol - model.solution
        return delta
