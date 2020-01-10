from scipy.optimize import linprog

from . import abstract_iterator
import numpy as np

from .tolsolvty.tolsolvty import tolsolvty
from ..data_parser.data_extractor import DataExtractor


class TosolvityIterator(abstract_iterator.AbstractIterator):
    """ML analog, which flattens arrays during calculation. Experimental feature.
    """

    def __init__(self):
        super().__init__(alpha=0.1, alpha_calc=None)
        self.matrix = []
        self.b_inf = []
        self.b_sup = []

    def init(self, model, *args, **kwargs):  # maybe make this __init__
        mfilename, gfilename = kwargs['mfile'], kwargs['gfile']

        data_extractor = DataExtractor(mfilename, gfilename)
        # huiny polpolycha
        self.matrix = np.zeros(256)
        #
        self.b_inf = []
        self.b_sup = []

    def finalize(self, model):
        model._detector_geometry = model.detector_geometry.reshape(self.det_shape)
        model._solution = model.solution.reshape(self.shape)

    @property
    def __str__(self):
        return 'Maximum Likelihood method'

    def step(self, model, step_num):
        model.solution = TosolvityIterator._solve(self.matrix, self.b_inf, self.b_sup)

    @staticmethod
    def _solve(a, b_inf, b_sup):
        """
        Function for solving interval linear system `ax = [b_inf, b_sup]`
        :param a: (M, N) array_like
            "coefficient" matrix
        :param b_inf: (M,) array_like
            ordinate or "dependent variable" values
        :param b_sup: (M,) array_like
            ordinate or "dependent variable" values
        :param args:
            see description for optional arguments for `tolsolvty` function
        :param verbose:
            if `True` it needs to draw plots
        :return: tolmax, argmax
            see description for returning values of `tolsolvty` function
        """

        tolmax, argmax, envs, ccode = tolsolvty(a, a, b_inf, b_sup)

        return tolmax, argmax
