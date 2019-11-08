from abc import ABC, abstractmethod

class AbstractMesh(ABC):
    """

    """
    @abstractmethod
    def __init__(self, dimensions=2, axis_names=None):
        self._dimensions - dimensions
        self.axis_names = axis_names

    # @property
    # def axis_names(self):
    #     return self._axis_names
    #
    # @axis_names.setter
    # def axis_names(self, value):
    #     if value is None:
    #         self._axis_names = None
    #     else:
    #         self.axis_names !=

    @abstractmethod
    def get_dimensions(self):
       """

       :return:
       """

    @abstractmethod
    def volume(self, cell_id):
        """

                      :return:
        """

    @abstractmethod
    def coordinates(self, cell_id):
        """

                      :return:
        """

    @abstractmethod
    def intersection_volumes(self, obj):
        """Calculate intersection volume of object with each mesh cell individually.

        Args:
            obj: object of particular format.

        Returns:
            dimensions-D Iterable of numbers. Represents of the same size as
        """

    @abstractmethod
    def draw(self, data, *args, **kwargs):
        """Visualize data on a mesh.

        Args:
            data: data to visualize - model solution or detector geometry.

        Returns:
            None
        """
