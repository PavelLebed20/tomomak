import warnings
import numpy
import shapely.geometry
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from tomomak.plots.plot2d import spiderweb_colormesh2d, detector_spiderweb_colormesh2d
from tomomak.mesh.abstract_axes import Abstract2dAxis


class SpiderWeb2dAxis(Abstract2dAxis):
    def __init__(self, border=None, center=None, radials_size=None, angle_size=None, name="", units=""):
        """
        Args:
            border(ndarray): array of points of separatrix
            center(ndarray): position of magnetic axis
            radials_size(int): number of radial lines that would be created (excluding separatrix)
            angle_size(int): number of sectors in one quarter of web (total angle axis size will be 'angle_size * 4')
        """

        super().__init__(name=name, units=units)

        if border is None:
            raise Exception("SpiderWeb2dAxis init: 'border' argument is None.")
        if center is None:
            raise Exception("SpiderWeb2dAxis init: 'center' argument is None.")

        if radials_size is None:
            warnings.warn("SpiderWeb2dAxis init: size of radials axis was not set. Default size = 3 is used.")
            self._radials_size = 3
        else:
            self._radials_size = radials_size

        if angle_size is None:
            warnings.warn("SpiderWeb2dAxis init: size of angle axis was not set. Default size = 4 is used.")
            self._angle_size = 4
        else:
            self._angle_size = angle_size

        self._name = name
        self._units = units

        self._angle_size = self._angle_size * 4
        self._border = border
        self._center = center

        self._create_domains()

    def _get_domains(self, border, center, radial_indices, count):
        """Function for getting all grid domains

        Args:
            border(ndarray): outer border line that represents by ndarrays; closed curve, but border[0] is not equal to border[-1]
            center(ndarray): two digits - point of the center of the grid
            radial_indices(set): radial indices including four support indices
            count(int): count of lines to generate; must be non-negative

        Returns:
            ndarray: generated grid domains
        """

        def get_indices():
            return radial_indices[i], radial_indices[(i + 1) % n]

        def get_two_points(a, b):
            return border[ind] * a + center * b, border[ind1] * a + center * b

        center = numpy.array(center)
        radial_indices = list(radial_indices)
        domains = []
        n = len(radial_indices)
        if count > 0:
            alpha = 1 / (count + 1)
            alpha_1 = 1 - alpha
            for i in range(n):
                ind, ind1 = get_indices()
                p2, p1 = get_two_points(alpha_1, alpha)
                domains.append(numpy.concatenate((border[ind:ind1 - 1], [border[ind1 - 1], border[ind1], p1, p2])))
            for k in range(1, count):
                ak = alpha * k
                ak_1 = 1 - ak
                ak1 = alpha * (k + 1)
                ak1_1 = 1 - ak1
                for i in range(n):
                    ind, ind1 = get_indices()
                    p1, p2 = get_two_points(ak_1, ak)
                    p4, p3 = get_two_points(ak1_1, ak1)
                    domains.append(numpy.array([p1, p2, p3, p4]))
            for i in range(n):
                ind, ind1 = get_indices()
                p1, p2 = get_two_points(alpha, alpha_1)
                domains.append(numpy.array([p1, p2, center]))
        else:
            for i in range(n):
                ind, ind1 = get_indices()
                domains.append(numpy.concatenate((border[ind:ind1 - 1], [border[ind1 - 1], border[ind1], center])))
        return numpy.array(domains)

    def _get4indices(self, line, radius):
        """Function for getting four indices of points for four support radial lines

        Args:
            line(ndarray): array of points that represent by ndarrays; closed curve, but line[0] is not equal to line[-1]
            radius(ndarray): array of curvature radii for each point of `line`

        Returns:
            ndarray: four indices in the `line` of four points for four support radial lines
        """

        n = len(radius)
        ind1, ind2 = numpy.nonzero(line[:, 1] == 0)[0]
        min1 = numpy.argmin(radius[ind1 + 1:ind2])
        min2 = numpy.argmin(numpy.concatenate((radius[ind2 + 1:n], radius[0:ind1])))
        return sorted([min1 + 1, (min2 + ind2 + 1) % n, ind1, ind2])

    def _get_radial_indices(self, border, indices, radials):
        """Function for getting all radial indices

        Args:
            border(ndarray): outer border line that represents by ndarrays; closed curve, but border[0] is not equal to border[-1]
            indices(ndarray): four indices in the `line` of four points for four support radial lines
            radials(int): count of radial lines between four support radial lines; must be non-negative

        Returns:
            set: generated radial indices including four support `indices`
        """

        radial_indices = set()
        indices_len = len(indices)
        n = len(border)
        for i in range(indices_len):
            ind1 = indices[i]
            ind2 = indices[(i + 1) % indices_len]
            if ind2 < ind1:
                ind1 = ind2 - (ind2 + n - ind1)
            for k in numpy.linspace(ind1, ind2, radials + 2):
                if not radial_indices.__contains__(n + k):
                    radial_indices.add(int(k))
        return radial_indices

    def _get_radius(self, line):
        """Function for getting curvature radius

        Args:
            line(ndarray): array of points that represent by ndarrays; closed curve, but line[0] is not equal to line[-1]

        Returns:
            ndarray: array of curvature radii for each point of `line`
        """

        n = len(line)
        radius = numpy.zeros(n)
        for i in range(n):
            p1 = line[i - 2]
            p2 = line[i - 1]
            p3 = line[i]
            m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
            m2 = (p3[1] - p2[1]) / (p3[0] - p2[0])
            x = (m1 * m2 * (p1[1] - p3[1]) + m2 * (p1[0] + p2[0]) - m1 * (p2[0] + p3[0])) / (2 * (m2 - m1))
            y = ((p1[0] + p2[0]) / 2 - x) / m1 + (p1[1] + p2[1]) / 2
            radius[i - 1] = numpy.linalg.norm(p1 - numpy.array([x, y]))
        return radius

    def _generate_domain_grid(self, border, center, count=0, radials=0):
        """Function for generation of grid

        Args:
            border(ndarray): outer border line that represents by ndarrays; closed curve, but border[0] is not equal to border[-1]
            center(ndarray): two digits - point of the center of the grid
            count(int): count of lines to generate; must be non-negative
            radials(int): count of radial lines between four support radial lines; must be non-negative

        Returns:
            ndarray: generated grid domains
        """

        radials = int(radials)
        count = int(count)
        radius = self._get_radius(border)
        indices = self._get4indices(border, radius)
        radial_indices = self._get_radial_indices(border, indices, radials)
        return self._get_domains(border, center, radial_indices, count)

    def _get_domains_array(self, border, center, count=0, radials=0):
        """Function for generation of grid and reordering domains of this grid

        Args:
            border(ndarray): outer border line that represents by ndarrays; closed curve, but border[0] is not equal to border[-1]
            center(ndarray): two digits - point of the center of the grid
            count(int): count of lines to generate; must be non-negative
            radials(int): count of radial lines between four support radial lines; must be non-negative

        Returns:
            ndarray: generated grid domains
        """

        domains = self._generate_domain_grid(border, center, count=count, radials=radials)
        n = (radials + 1) * 4
        for i in range(count + 1):
            up_domains = numpy.flip(domains[i * n:i * n + n // 2], axis=0)
            bottom_domains = numpy.flip(domains[i * n + n // 2:(i + 1) * n], axis=0)
            domains[i * n:(i + 1) * n] = numpy.concatenate((up_domains, bottom_domains))
        return domains

    def _create_domains(self):
        self._domains = self._get_domains_array(border=self._border, center=self._center,
                                                count=self._radials_size, radials=self._angle_size)

    @property
    def cell_edges(self):
        res = []
        for i in range(self._radials_size):
            for j in range(self._angle_size):
                res.append((i, j))

        return res

    @property
    def volumes(self):
        res = numpy.zeros((len(self._domains), 2))
        for i, domain in enumerate(self._domains):
            res[i] = shapely.geometry.Polygon(domain).area
        return res

    @property
    def coordinates(self):
        res = numpy.zeros((len(self._domains), 2))
        for i, domain in enumerate(self._domains):
            res[i] = shapely.geometry.Polygon(domain).centroid
        return res

    @property
    def size(self):
        return len(self._domains)

    @property
    def regular(self):
        return False

    def cell_edges2d(self):
        return self._domains

    def intersection(self, axis2):
        pass

    def plot2d(self, data, data_type='solution', fill_scheme='viridis', grid=False, *args, **kwargs):
        if data_type == 'solution':
            title = r"Density, {}".format(self._units)
            plot, ax, fig, cb = spiderweb_colormesh2d(data, self, title, fill_scheme, grid, *args, **kwargs)
        elif data_type == 'detector_geometry':
            title = 'Detector 1/{}'.format(data.shape[0])
            plot, ax, _ = detector_spiderweb_colormesh2d(data, self, title, fill_scheme, grid, *args, **kwargs)
        else:
            raise AttributeError('data type {} is unknown'.format(data_type))
        plt.show()
        return plot, ax

    def plot3d(self, data, axis2, *args, **kwargs):
        raise NotImplemented("SpiderWeb2dAxis plot3d: not implemented method")

    def cell_edges3d(self, axis2):
        if axis2.coordinates[0] < 0 or axis2.coordinates[-1] > 2 * numpy.pi:
            raise AttributeError('axis2 is not like circled axis (lower_limit < 0 or axis2.upper_limit > 2 * pi)')

        shape = (self.size, axis2.size)
        res = numpy.zeros(shape).tolist()
        step = 1
        for i, coord in enumerate(axis2.coordinates):
            if i == len(axis2.coordinates) - 1:
                step = -i
            coord_next = axis2.coordinates[i + step]
            for j, domain in enumerate(self._domains):
                far_rotated_domain = []
                near_rotated_domain = []

                r = R.from_euler('y', coord_next, degrees=False)
                for point in domain:
                    far_rotated_domain.append(r.apply((point[0], point[1], 0)))

                r = R.from_euler('y', coord, degrees=False)
                for point in domain:
                    near_rotated_domain.append(r.apply((point[0], point[1], 0)))

                r = near_rotated_domain
                r.extend(far_rotated_domain)
                res[j][i] = r

        return res
