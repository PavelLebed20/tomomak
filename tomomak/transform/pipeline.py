"""Module, defining transformation pipeline class and abstract transformer.
"""
from abc import ABC, abstractmethod


class Pipeline:
    """Pipeline, providing interface for  model transformation.

        Pipeline is a top-level container for all transforms. It is one of the core TOMOMAK structures.
        Pipeline is usually used to perform a number of transformations before calculation of the solution
        and inverse transformation after solution is obtained.
        Object can be used as a transformer if it implements forward() and backward() methods
        and is able to store Model object. Model should be changed inside of these methods.
        """

    def __init__(self, model, transformers=()):
        self._model = model
        self._position = 0
        self._len = 0
        self._transformers = []
        for t in transformers:
            self._len += 1
            self._transformers.append(t)

    @property
    def position(self):
        return self._position

    def add_transform(self, transformer):
        self._transformers.append(transformer)
        self._len += 1

    def remove_transform(self, index):
        del self._transformers[index]
        self._len -= 1

    def forward(self, steps=1):
        if steps < 0:
            raise TypeError("Number of steps should be positive.")
        if self._position + steps > self._len + 1:
            raise Exception("Unable to perform transformation since final index is out of range.")
        for i in range(steps):
            self._transformers[self._position].forward()
            self._position += 1

    def backward(self, steps=1):
        if steps < 0:
            raise TypeError("Number of steps should be positive.")
        if self._position - steps < 0:
            raise Exception("Unable to perform transformation since final index is out of range.")
        for i in range(steps):
            self._transformers[self._position - 1].backward()
            self._position -= 1

    def to_last(self):
        steps = self._len - self._position
        self.forward(steps)

    def to_first(self):
        steps = self._position
        self.backward(steps)


class AbstractTransformer(ABC):
    """
    
    """
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def forward(self):
        """

        """

    @abstractmethod
    def backward(self):
        """

        """