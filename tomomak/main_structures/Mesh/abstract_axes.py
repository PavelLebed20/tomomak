from abc import ABC, abstractmethod


class Abstract1dAxis(ABC):

    def __init__(self, name="", units=""):
        self.name = name
        self.units = units

    @property
    def dimension(self):
        return 1

    @abstractmethod
    def plot1d(self, data, *args, **kwargs):
        """

        :return:
        """

    @property
    @abstractmethod
    def volumes(self):
        """

        Returns:

        """

    @property
    @abstractmethod
    def coordinates(self):
        """

        Returns:

        """

    @property
    @abstractmethod
    def boundaries(self):
        """

        :return:
        """

    @property
    @abstractmethod
    def cell_edges(self):
        """

        :return:
        """

    @abstractmethod
    def to2d(self, axis2):
        """

        :param axis2:
        :return:
        """
