import numpy as np


def _extract(filename, mag_mesh):
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


def gfile_extract(filename):
    mag_mesh = 65
    flux, RBDRY, ZBDRY, NBDRY, R, Z, rdim, zdim = _extract(filename, mag_mesh)
    min = np.argmin(flux)
    zmin = min // mag_mesh
    rmin = min - zmin * mag_mesh
    center = (R[rmin], Z[zmin])
    border = np.column_stack((RBDRY, ZBDRY))
    return border, center