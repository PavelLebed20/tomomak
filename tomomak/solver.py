import numpy as np
import numbers
import warnings
import copy

class Solver:
    """
    Args:
        alpha(float of iterable of floats):
    """
    def __init__(self, iterator=None, alpha=0.1, alpha_calc=None, constraints_array=None, stat_array=None,
                 stop_criteria=None, real_solution=None):
        self.iterator = iterator
        self.alpha = alpha
        self.alpha_calc = alpha_calc
        self.constraints_array = constraints_array
        self.stat_array = stat_array
        self.stop_criteria = stop_criteria
        self.real_solution = real_solution
        self.statistics = []

    def solve(self, model, steps=20, *args, **kwargs):
        # Check consistency.
        if model.detector_signal is None:
            raise ValueError("detector_signal should be defined to perform reconstruction.")
        if model.detector_geometry is None:
            raise ValueError("detector_geometry should be defined to perform reconstruction.")
        # Init iterator and constraints.
        print("Start calculation with {} iterations using {}.".format(steps, self.iterator))
        self.iterator.init(model, steps, *args, **kwargs)
        if self.constraints_array is not None:
            for r in self.constraints_array:
                r.init(model, steps, *args, **kwargs)
            print("Used constraints:")
            for c in self.constraints_array:
                print(c)
        if self.stop_criteria is not None:
            self.stop_criteria .init(model, steps, *args, **kwargs)
            print("Stopping criteria: {}".format(self.stop_criteria))
        if self.alpha_calc is not None:
            self.alpha_calc.init(model, steps, *args, **kwargs)
            print("Method of step calculation: {}".format(self.alpha_calc))
        # Start iteration
        for i in range(steps):
            old_solution = copy.copy(model.solution)
            self.iterator.step(model=model, step_num=i)
            # constraints
            if self.constraints_array is not None:
                for k, r in enumerate(self.constraints_array):
                    r.step(model=model, step_num=i)
            # statistics
            if self.stat_array is not None:
                step_statistic = []
                for s in self.stat_array:
                    s(solution=model.solution, step_num=i, real_solution=self.real_solution,
                      old_solution=old_solution, model=model)
                self.statistics.append(step_statistic)
            # early stopping
            if self.stop_criteria is not None:
                pass ######## break
            if i % 20 == 0:
                print('\r', end='')
                print("...", str(i * 100 // steps) + "% complete", end='')

        print('\r \r', end='')
        self.iterator.finalize(model)
        print("Calculation ended in ", i + 1, "steps.")
