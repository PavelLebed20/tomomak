import numpy as np
import numbers


class Solver:
    """
    Args:
        alpha(float of iterable of floats):
    """
    def __init__(self, iterator=None, alpha = 0.1, reg_array=None, reg_alpha = None, stat_array=None):
        self.iterator = iterator
        self.alpha = alpha
        self.reg_array = reg_array
        self.reg_alpha = reg_alpha
        self.stat_array = stat_array
        self.statistics = []

    def solve(self, model, steps=20,  stop_type='None', stop_val=0, real_solution=None, *args, **kwargs):
        if model.detector_signal is None:
            raise ValueError("detector_signal should be defined to perform reconstruction.")
        if model.detector_geometry is None:
            raise ValueError("detector_geometry should be defined to perform reconstruction.")
        print("Start calculation with {} iterations using {}.".format(steps, self.iterator.name))
        ### if not alpha calc
        if isinstance(self.alpha, numbers.Number):
            alpha = np.full(steps, self.alpha)
        else:
            alpha = self.alpha
        if len(alpha) != steps:
            raise ValueError("Alpha len should be equal to steps.")
        self.iterator.init(model, *args, **kwargs)
        sol = model.solution
        for i in range(steps):
            # iterator gradient
            delta = self.iterator.step(model) ####
            # regularisation gradient
            delta_reg = 0 #####
            #######calc alphas
            if self.reg_array is not None:
                for i, r in enumerate(self._reg_array):
                    dr = r ############
                    delta_reg += self.reg_alpha[i] * dr
            # get next_value
            new_sol = model.solution + alpha[i] * delta + delta_reg
            # statistics
            if self.stat_array is not None:
                step_statistic = []
                for s in self.stat_array:
                    s(solution=sol, real_solution = real_solution, old_solution=model.solution, model=model)
                self.statistics.append(step_statistic)
            model.solution = new_sol
            # early stopping
            if stop_type is not None:
                pass ########
            if i % 20 == 0:
                print('\r', end='')
                print("...", str(i * 100 // steps) + "% complete", end='')
        print('\r \r', end='')
        self.iterator.finalize(model)
        print("Calculation ended in ", i + 1, "steps.")
