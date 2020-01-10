import numpy as np
from scipy.io import loadmat

class DataExtractor:
    def __init__(self, mfilename, gfilename, mag_mesh=65):
        """
        Initialize data extractor class
        """
        self.flux, self.RBDRY, self.ZBDRY, \
            self.NBDRY, self.R, self.Z, self.rdim, self.zdim = self.extract(gfilename, mag_mesh)

        self.min = np.argmin(self.flux)
        self.zmin = self.min // mag_mesh
        self.rmin = self.min - self.zmin * mag_mesh
        self.enter = (self.R[self.rmin], self.Z[self.zmin])
        self.border = np.column_stack((self.RBDRY, self.ZBDRY))

        mat = loadmat(mfilename)
        sign_bb = mat['sign_bb']
        tp = mat['Data'][0][1][0][0] * 1e-3
        tz = mat['Data'][1][1][0][0]
        ind = int((t - tz) / tp)
        ind_inf = ind - 1 if ind > 0 else ind
        ind_sup = ind + 1 if ind < sign_bb.shape[2] - 1 else ind
        b_inf = np.min(sign_bb[:, :, ind_inf:ind_sup + 1], axis=2)
        b_sup = np.max(sign_bb[:, :, ind_inf:ind_sup + 1], axis=2)
        b_inf = np.rot90(b_inf, 2).T.reshape(256)
        b_sup = np.rot90(b_sup, 2).T.reshape(256)
        self.b_inf, self.b_sup = b_inf, b_sup

    @staticmethod
    def extract(filename, mag_mesh):
        """
        Extract data from gfile
        :param filename: gfile path
        :param mag_mesh: mesh size
        :return:
        """
        data = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                data.extend(line.split())
        rdim = float(data[9])  # размер сетки по радиусу в метрах
        zdim = float(data[10])  # размер сетки по Z в метрах

        delay = 15 + 55 * 5 - 1
        flux = np.zeros((mag_mesh, mag_mesh))
        for i in range(mag_mesh):
            for j in range(mag_mesh):
                flux[i, j] = float(data[delay])
                delay = delay + 1

        NBDRY = 0  # количество точек сепаратриссы
        for i in range(len(data)):
            if data[i] == 'NBDRY':
                NBDRY = int(data[i + 2])
                break

        RBDRY = np.zeros(NBDRY)  # координаты сепаратриссы по радиусу
        for i in range(len(data)):
            if data[i] == 'RBDRY':
                i += 2
                for k in range(NBDRY):
                    RBDRY[k] = float(data[i + k])

        ZBDRY = np.zeros(NBDRY)  # координаты сепаратриссы по Z
        for i in range(len(data)):
            if data[i] == 'ZBDRY':
                i += 2
                for k in range(NBDRY):
                    ZBDRY[k] = float(data[i + k])

        Z = 0.5 * zdim * np.arange(-mag_mesh + 1, mag_mesh, 2) / mag_mesh  # расчитываем координатную сетку в метрах
        R = 0.5 * rdim * np.arange(1, 2 * mag_mesh, 2) / mag_mesh

        return flux, RBDRY, ZBDRY, NBDRY, R, Z, rdim, zdim
