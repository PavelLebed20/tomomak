from . import abstract_iterator
import numpy as np
import warnings


class ML(abstract_iterator.AbstractIterator):

    def __init__(self):
        self.w_det = None
        self.wi = None
        self.shape = None

    def init(self, model): #maybe make this __init__
        if model.solution is None:
            shape = model.mesh.shape
            model.solution = np.ones(shape)
        else:
            if np.all(model.solution):
                warnings.warn("Some elements in model solution are zero. They will not be changed.")
        self.shape = model.solution.shape
        self.w_det = np.multiply(np.moveaxis(model.detector_geometry, 0, -1), model.detector_signal)
        self.wi = np.sum(model.detector_geometry, axis=0)

    def finalize(self, model):
        pass

    @property
    def name(self):
        return 'Maximum Likelihood method'

    def step(self, model):
        # expected signal
        y_expected = np.zeros(model.detector_signal.shape)
        for i, geom in enumerate(model.detector_geometry):
            y_expected[i] = np.sum(np.multiply(model.solution, geom))
        # multiplication
        mult = np.sum(np.divide(self.w_det, y_expected, out=np.zeros_like(self.w_det), where=y_expected != 0), axis=-1)
        mult = mult / self.wi
        # find delta
        new_sol = model.solution * mult
        delta = new_sol - model.solution
        return delta


