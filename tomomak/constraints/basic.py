from ..iterators import abstract_iterator
import numpy as np
import warnings
from tomomak.detectors import signal


class Positive(abstract_iterator.AbstractIterator):

    def __init__(self):
        super().__init__(None, None)

    def init(self, model, steps, *args, **kwargs):
        pass

    def finalize(self, model):
        pass

    def __str__(self):
        return "Remove negative values"

    def step(self, model, step_num):
        model.solution = model.solution.clip(min=0)

