import numbers


class Model:
    """
    1 axis = 1 solution array dimension
    """

    def __init__(self, detector_geometry=None, detector_signal=None, solution=None, mesh=None):
        self._detector_geometry = detector_geometry
        self._detector_signal = detector_signal
        self._solution = solution
        self._mesh = mesh
        self._check_self_consistency()

    def __str__(self):
        notdef = "Not defined."
        res = "Model description:\nNumber of detectors: "
        if self._detector_signal is not None:
            n_det = str(len(self._detector_signal))
        elif self._detector_geometry is not None:
            n_det = str(self._detector_geometry.shape[0])
        else:
            n_det = notdef
        res += n_det
        if self.detector_geometry is not None:
            n_cells = str(self.detector_geometry[0].size)
        elif self._mesh is not None:
            n_cells = 0
            for n in self._mesh:
                n_cells += n.size
            n_cells = str(n_cells)
        else:
            n_cells = notdef
        res += "\nNumber of cells: {}\nMesh:\n".format(n_cells)
        if self._mesh is not None:
            mesh = str(self._mesh)
        else:
            mesh = notdef
        res += mesh
        res += "Solution: "
        if self.solution is not None:
            solution = "Defined"
        else:
            solution = notdef
        res += solution
        return res

    @property
    def detector_geometry(self):
        return self._detector_geometry

    @detector_geometry.setter
    def detector_geometry(self, value):
        self._detector_geometry = value
        self._check_self_consistency()

    @property
    def detector_signal(self):
        return self._detector_signal

    @detector_signal.setter
    def detector_signal(self, value):
        self._detector_signal = value
        self._check_self_consistency()

    @property
    def solution(self):
        return self._solution

    @solution.setter
    def solution(self, value):
        self._solution = value
        self._check_self_consistency()

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value
        self._check_self_consistency()

    def _check_self_consistency(self):
        """
        Check model self-consistency.
        Self-consistency is checked if an attribute is changed.
        """
        if self._detector_geometry is not None:
            geometry_len = len(self._detector_geometry)
            if self._detector_signal is not None:
                if not isinstance(self._detector_signal[0], numbers.Number):
                    raise TypeError("detector_signal should be 1D iterable of numbers")
                signal_len = len(self._detector_signal)
                if geometry_len != signal_len:
                    raise Exception("detector_signal and detector_geometry should have same length. "
                                    "detector_geometry len is {}; detector signal len is {}."
                                    .format(geometry_len, signal_len))
            if self._solution is not None:
                if self._solution.shape != self._detector_geometry[0].shape:
                    raise Exception("Each slice in detector_geometry should have same shape as solution. "
                                    "detector_geometry[0] shape is {}; solution shape is {}."
                                    .format(self._detector_geometry[0].shape, self._solution.shape))
        if self._mesh is not None:
            for val, name in zip([self.detector_geometry[0], self.solution], ["detector_geometry", "solution"]):
                if val is not None:
                    # if self.mesh.dimension != len(val.shape):
                    #     raise Exception("Mesh dimension is inconsistent with {}. Mesh is {}-D while {} is {}-D."
                    #                     .format(name, self.mesh.dimension, name, len(val.shape)))
                    if self.mesh.shape != val.shape:
                        raise Exception("Mesh shape is inconsistent with {}. Mesh shape is {} while {} is {}."
                                        .format(name, self.mesh.shape, name, val.shape))



    def plot1d(self, index=0, data_type="solution", **kwargs):
        if data_type == "solution":
            data = self._solution
        elif data_type == "detector_geometry":
            data = self._detector_geometry
        else:
            raise AttributeError("data type {} is unknown".format(data_type))
        plot, ax = self._mesh.plot1d(data, index, data_type,  **kwargs)
        return plot, ax

    def plot2d(self, index=0, data_type="solution", **kwargs):
        if data_type == "solution":
            data = self._solution
        elif data_type == "detector_geometry":
            data = self._detector_geometry
        else:
            raise AttributeError("data type {} is unknown".format(data_type))
        plot, ax = self._mesh.plot2d(data, index, data_type,  **kwargs)
        return plot, ax