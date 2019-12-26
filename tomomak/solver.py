import numpy as np
import numbers
import warnings
import copy


class Solver:
    """
    Args:

    """
    def __init__(self, iterator=None, constraints_array=None, stat_array=None,
                 stop_array=None, stop_values=None, real_solution=None):
        self.iterator = iterator
        self.constraints_array = constraints_array
        self.stat_array = stat_array
        self.stop_values = stop_values
        self.stop_array= stop_array
        self.real_solution = real_solution
        self.statistics = []

    def solve(self, model, steps=20, *args, **kwargs):
        # Check consistency.
        if model.detector_signal is None:
            raise ValueError("detector_signal should be defined to perform reconstruction.")
        if model.detector_geometry is None:
            raise ValueError("detector_geometry should be defined to perform reconstruction.")
        if self.stop_array is not None:
            if self.stop_values is None:
                raise ValueError("stop_values should be defined since stop_array is defined.")
            if len(self.stop_values) != len(self.stop_array):
                raise ValueError("stop_array and stop_values have different length.")
        # Init iterator and constraints.
        print("Start calculation with {} iterations using {}.".format(steps, self.iterator))
        if self.iterator is not None:
            self.iterator.init(model, steps, *args, **kwargs)
        if self.constraints_array is not None:
            for r in self.constraints_array:
                r.init(model, steps, *args, **kwargs)
            print("Used constraints:")
            for c in self.constraints_array:
                print(c)

        # Start iteration
        for i in range(steps):
            old_solution = copy.copy(model.solution)
            if self.iterator is not None:
                self.iterator.step(model=model, step_num=i)
            # constraints
            if self.constraints_array is not None:
                for k, r in enumerate(self.constraints_array):
                    r.step(model=model, step_num=i)
            # statistics
            if self.stat_array is not None:
                step_statistic = []
                for s in self.stat_array:
                    step_statistic.append(s(solution=model.solution, step_num=i, real_solution=self.real_solution,
                                            old_solution=old_solution, model=model))
                self.statistics.append(step_statistic)
            # early stopping
            if self.stop_array is not None:
                stop = False
                for k, s in enumerate(self.stop_array):
                    val = s(solution=model.solution, step_num=i, real_solution=self.real_solution,
                            old_solution=old_solution, model=model)
                    if val < self.stop_values[k]:
                        print('\r \r', end='')
                        print("Early stopping: " + s.__name__ + " < " + str(self.stop_values[k]))
                        stop = True
                if stop:
                    break
            if i % 20 == 0:
                print('\r', end='')
                print("...", str(i * 100 // steps) + "% complete", end='')

        print('\r \r', end='')
        if self.iterator is not None:
            self.iterator.finalize(model)
        if self.constraints_array is not None:
            for r in self.constraints_array:
                r.finalize(model)

