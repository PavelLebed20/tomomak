from .pipeline import AbstractTransformer


class Rescaler(AbstractTransformer):
    """
    Rescales to the new shape. Keeps proportions for irregular grid. Doesn't smooth.
    """
    def __init__(self, new_shape):
        self.new_shape = new_shape
        self.old_shape = None

    @staticmethod
    def _rescale(new_shape, model):
        if model.mesh is None:
            raise Exception("Unable to rescale model with undefined Mesh.")
        if len(model.mesh.axes) != len(new_shape):
            raise Exception("Number of the new shape axes should be equal to number of Mesh axes. "
                            "New shape has {} axes. Mesh has {} axes".format(len(new_shape), model.mesh.axes))
        for i, ax in enumerate(model.mesh.axes):
            new_len = new_shape[i]
            old_len = ax.size
            ratio = new_len / old_len




    def forward(self,  model):
        old_shape = model.shape
        self._rescale(self.new_shape, model)
        self.old_shape = old_shape


    def backward(self,  model):
        self._rescale(self.old_shape, model)