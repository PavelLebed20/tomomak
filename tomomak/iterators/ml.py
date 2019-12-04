from . import abstract_iterator
import numpy as np
import warnings
from tomomak.detectors import signal

class ML(abstract_iterator.AbstractIterator):
    """Maximum likelihood iterative solver for image reconstruction
    see  for example G. Kontaxakis and L.G. Strauss
    - Maximum Likelihood Algorithms for Image Reconstruction in Positron Emission Tomography.
    All attributes and methods are used automatically in solver (see tomomak.iterators.abstract_iterator).
    """

    def __init__(self):
        self.w_det = None
        self.wi = None
        self.shape = None

    def init(self, model):  # maybe make this __init__
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
        y_expected = signal.get_signal(model.solution, model.detector_geometry)
        # multiplication
        mult = np.sum(np.divide(self.w_det, y_expected, out=np.zeros_like(self.w_det), where=y_expected != 0), axis=-1)
        mult = mult / self.wi
        # find delta
        new_sol = model.solution * mult
        delta = new_sol - model.solution
        return delta


class MLFlatten(abstract_iterator.AbstractIterator):
    """ML analog, which flattens arrays during calculation.
    """

    def __init__(self):
        self.w_det = None
        self.wi = None
        self.shape = None
        self.det_shape = None

    def init(self, model):  # maybe make this __init__
        if model.solution is None:
            shape = model.mesh.shape
            model.solution = np.ones(shape)
        else:
            if np.all(model.solution):
                warnings.warn("Some elements in model solution are zero. They will not be changed.")
        self.shape = model.solution.shape
        self.det_shape = model.detector_geometry.shape
        model._solution = model.solution.flatten()
        shape2 = np.prod(self.shape)
        model._detector_geometry = model.detector_geometry.reshape((self.det_shape[0], shape2))
        self.w_det = np.multiply(np.moveaxis(model.detector_geometry, 0, -1), model.detector_signal)
        self.wi = np.sum(model.detector_geometry, axis=0)

    def finalize(self, model):
        model._detector_geometry = model.detector_geometry.reshape(self.det_shape)
        model._solution = model.solution.reshape(self.shape)

    @property
    def name(self):
        return 'Maximum Likelihood method'

    def step(self, model):
        # expected signal
        y_expected = signal.get_signal(model.solution, model.detector_geometry)
        # multiplication
        mult = np.sum(np.divide(self.w_det, y_expected, out=np.zeros_like(self.w_det), where=y_expected != 0), axis=-1)
        mult = mult / self.wi
        # find delta
        new_sol = model.solution * mult
        delta = new_sol - model.solution
        return delta
