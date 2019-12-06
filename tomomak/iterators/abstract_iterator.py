from abc import ABC, abstractmethod
import warnings
import numbers
import numpy as np


class AbstractIterator(ABC):
    """
    """
    def __init__(self, alpha=0.1, alpha_calc=None):
        self.alpha = alpha
        self.alpha_calc = alpha_calc
        self._alpha = None

    @abstractmethod
    def init(self, model, steps, *args, **kwargs): ###NEED test for this
        """Use this (super().init(model, steps, *args, **kwargs)) to enable adding list of alphas.
        """
        if self.alpha_calc is not None:
            if self.alpha is not None:
                self._alpha = None
                warnings.warn("Since alpha_calc is defined in {}, alpha is Ignored.".format(self))
        else:
            if isinstance(self.alpha, numbers.Number):
                self._alpha = np.full(steps, self.alpha)
            else:
                self._alpha = self.alpha
            if len(self._alpha) < steps:
                raise ValueError("Alpha len in {} should be equal or greater than number of steps.".format(self))
        """

        :return:
        """

    @abstractmethod
    def finalize(self, model):

        """

        Returns:

        """

    def get_alpha(self, model, step_num):
        """Use this to get alpha.
        """
        if self.alpha_calc is not None:
            alpha = self.alpha_calc.step(model=model)
        else:
            alpha = self._alpha[step_num]
        return alpha

    @abstractmethod
    def step(self, model, step_num):
        """Use this (alpha = super().step(model, step_num)) to get alpha/

        Args:
            model:
            step_num

        Returns:
            None

        """


    @abstractmethod
    def __str__(self):
        """Return name or name with parameters.

        Returns:
            str:

        """
