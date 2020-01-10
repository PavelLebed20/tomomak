from scipy.optimize import linprog

from . import abstract_iterator
import numpy as np
from ..data_parser.data_extractor import DataExtractor


class LinprogIterator(abstract_iterator.AbstractIterator):
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
        model.solution = LinprogIterator._solve(self.matrix, self.b_inf, self.b_sup)

    @staticmethod
    def _solve(a, b_inf, b_sup):
        """
        Function for solving interval linear system `ax = [b_inf, b_sup]` like linear programming problem
        :param a: (M, N) array_like
            "coefficient" matrix
        :param b_inf: (M,) array_like
            ordinate or "dependent variable" values
        :param b_sup: (M,) array_like
            ordinate or "dependent variable" values
        :param verbose:
            if `True` it needs to draw plots
        :return: x: {(N,) ndarray, None}
            the independent variable vector which optimizes the linear programming problem
        :return: success: bool
            returns `True` if the algorithm succeeded in finding an optimal solution
        """

        a = np.array(a)
        b_inf = np.array(b_inf)
        b_sup = np.array(b_sup)

        m, n = a.shape[:2]
        ki = b_inf.shape[0]
        ks = b_sup.shape[0]
        if ki == ks:
            k = ks
        else:
            print('The number of components in the vectors of the left and right ends is not the same')
            return None, False
        if k != m:
            print('The dimensions of the system matrix do not match the dimensions of the right side')
            return None, False
        if not np.all(b_inf <= b_sup):
            print('Invalid interval component was set on the right side vector')
            return None, False

        f = np.concatenate((np.zeros(n), np.ones(k)))
        mid = (b_inf + b_sup) / 2
        rad = (b_sup - b_inf) / 2
        diag_rad = np.diag(rad)
        c = np.vstack((np.hstack((a, -diag_rad)), np.hstack((-a, -diag_rad))))
        d = np.concatenate((mid, -mid))

        res = linprog(f, A_ub=c, b_ub=d, options={"disp": True})
        z = res['x']
        success = res['success']
        x, w = z[:n], z[n:]
        print("function(z) = %f" % res['fun'])

        return x, success
