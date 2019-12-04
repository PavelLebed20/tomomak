from abc import ABC, abstractmethod


class AbstractIterator(ABC):

    @abstractmethod
    def init(self, model):
        """

        :return:
        """

    @abstractmethod
    def finalize(self, model):
        """

        Returns:

        """

    @abstractmethod
    def step(self, model):
        """

        Args:
            model:

        Returns:

        """
    @property
    @abstractmethod
    def name(self):
        """

        Returns:
            str:

        """
