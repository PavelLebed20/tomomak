import numbers


class Model:
    """

    """

    def __init__(self, detector_geometry=None, detector_signal=None, solution=None, mesh=None):
        self._detector_geometry = detector_geometry
        self._detector_signal = detector_signal
        self._solution = solution
        self._mesh = mesh
        self._check_self_consistency()

    @property
    def detector_geometry(self):
        return self._detector_geometry

    @detector_geometry.setter
    def detector_geometry(self, value):
        if value is None:
            self._detector_geometry = None
        else:
            self._check_self_consistency()

    @property
    def detector_signal(self):
        return self._detector_signal

    @detector_signal.setter
    def detector_signal(self, value):
        if value is None:
            self._detector_signal = None
        else:
            self._check_self_consistency()

    @property
    def solution(self):
        return self._solution

    @detector_signal.setter
    def detector_signal(self, value):
        if value is None:
            self._solution = None
        else:
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
################ADD MESH check################

    def plot1d(self, index=0, data_type="solution", **kwargs):
        if data_type == "solution":
            data = self._solution
        elif data_type == "detector_geometry":
            data = self._detector_geometry
        else:
            raise AttributeError("data type {} is unknown".format(data_type))
        self._mesh.plot1d(data=data, index=index, data_type=data_type,  **kwargs)

