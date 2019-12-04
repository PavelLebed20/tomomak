import numpy as np

def positive(solution, *args, **kwargs):
    positive_solution = solution.clip(min=0)
    return positive_solution - solution
