import numpy as np
import shapely.geometry
from numpy.linalg import norm

from tomomak.util.geometry.AbstractGeometry import AbstractGeometry


class Geometry3d(AbstractGeometry):

    @staticmethod
    def get_perp_to_line(p1, p2):
        dir = np.array(p1) - np.array(p2)
        T = np.array([dir[2], dir[2], -dir[0] - dir[1]])

        if T[0] == 0 and T[2] == 0:
            T = np.array([-dir[1] - dir[2], dir[0], dir[0]])

        return T / norm(T)

    @staticmethod
    def _line_to_polygon(p1, p2, width, divergence=0):
        """Generate detector geometry for one Line of Sight.

        line of sight can be collimated or diverging (cone-like).

        Args:
            p1, p2(tuple of 2 points):
            width(float): line width.
            divergence(float, optional): Angle between two LOS borders in Rad [0, pi). 0 means that line is collimated.
                default: 0.

        Returns:
            tuple of 6 points: List of 4 Shapely points, defining LOS.

        Raises:
            ValueError if divergence is < 0 or >= pi.
        """
        if divergence < 0 or divergence >= np.pi:
            raise ValueError("Divergence value is {}. It should be >= 0 and < pi.".format(divergence))
        # increase line length in case of line rotation
        p1 = np.array(p1)
        p2 = np.array(p2)
        r = p2 - p1
        r = r / np.cos(divergence)
        p2 = p1 + r
        # take into account width and divergence
        perp = Geometry3d.get_perp_to_line(p1, p2) * (width / 2)
        perp1 = np.cross(perp, r)
        perp1 = perp1 * (width / 2 / norm(perp1))

        points = np.zeros((8, 3))

        points[0] = p1 + perp
        points[1] = p1 - perp
        points[2] = p2 + perp
        points[3] = p2 - perp

        return points

    @staticmethod
    def cell_distances(mesh, index, p):
        """Get distance to each cell on mesh.

        Args:
            mesh(tomomak.main_structures.Mesh): mesh to work with.
            index(tuple of two ints, optional): axes to calculate distance at. Default:  (0,1)

        Returns:
            ndarray: 2D or 1D ndarray with distances.
            """
        p1 = shapely.geometry.Point(p)
        # If axis is 2D
        if len(index) == 2:
            # check one is 2 dimension, other is one
            i1 = index[0]
            i2 = index[1]
            if mesh.axes[i1].dimension != 2:
                i1, i2 = i2, i1
            assert mesh.axes[i1].dimension == 2
            assert mesh.axes[i2].dimension == 1
            shape = (mesh.axes[i1].size, mesh.axes[i2].size)
            r = np.zeros(shape)
            for i, row in enumerate(r):
                for j, _ in enumerate(row):
                    t1 = mesh.axes[i1].coordinates[i]
                    t2 = mesh.axes[i2].coordinates[j]

                    p2 = shapely.geometry.Point(t1[0], t1[1], t2[0])
                    r[i, j] = p1.distance(p2)
        # If axes are 1D
        elif len(index) == 3:
            i1 = index[0]
            i2 = index[1]
            i3 = index[2]
            for i in range(0, 3):
                assert mesh.axes[i].dimension == 1
            shape = (mesh.axes[i1].size, mesh.axes[i2].size, mesh.axes[i3].size)
            r = np.zeros(shape)
            for k, plane in enumerate(r):
                for i, row in enumerate(plane):
                    for j, _ in enumerate(row):
                        p2 = shapely.geometry.Point(mesh.axes[i1].coordinates[k], mesh.axes[i2].coordinates[i], mesh.axes[i3].coordinates[j])
                        r[k, i, j] = p1.distance(p2)
        return r

    @staticmethod
    def intersection(mesh, points, index=(0, 1, 2), calc_area=True):
        """Create solution array, representing 2d polygon, defined by specified points on the given mesh.

        If there are more than 3 dimension in model, broadcasting to other dimensions is performed.
        If broadcasting is not needed private method _polygon may be used.
        Only axes which implements cell_edges3d method are supported.
        cell_edges3d method should  accept second axe and return 2d list of ordered sequence of point tuples for three 1d axes
        or 1d list of ordered sequence of point tuples for one 3d axis.
        Each point tuple represents cell borders in the 3D cartesian coordinates.
        Shapely module is used for the calculation.

        Args:
            mesh(tomomak.main_structures.Mesh): mesh to work with.
            points(An ordered sequence of point tuples, optional): Polygon points (x, y, z)
            index(tuple of 3 ints, optional): axes to build object at. Default:  (0,1,2)
            calc_area(bool): If True, area of intersection with each cell is calculated, if False,
                only fact of intersecting with mesh cell is taken into account. Default: True.

        Returns:
            ndarray: 2D numpy array, representing polygon on the given mesh.

        Raises:
            TypeError if one of the axes is not  cartesian (tomomak.main_structures.mesh.cartesian).
        """
        if isinstance(index, int):
            index = [index]
        elif not isinstance(index, list) and not isinstance(index, tuple):
            index = [0, 1]

        tmp_p = np.array([points[0], points[7], points[1]])
        pol = shapely.geometry.Polygon(tmp_p)

        if len(index) < 2:
            raise Exception("Custom axis should implement cell_edges3d method. "
                            "This method returns 3d list of ordered sequence of point tuples."
                            " See docstring for more information.")
        # If axis is 2D
        if len(index) == 2:
            # check one is 2 dimension, other is one
            i1 = index[0]
            i2 = index[1]
            if mesh.axes[i1].dimension != 2:
                i1, i2 = i2, i1
            assert mesh.axes[i1].dimension == 2
            try:
                cells = mesh.axes[i1].cell_edges3d(mesh.axes[i2])
            except (TypeError, AttributeError) as e:
                raise type(e)(e.message + "Custom axis should implement cell_edges3d method. "
                                          "This method returns 3d list of ordered sequence of point tuples."
                                          " See docstring for more information.")
            shape = (mesh.axes[i1].size, mesh.axes[i2].size)
            res = np.zeros(shape)
            for i, row in enumerate(res):
                for j, _ in enumerate(row):
                    cell = shapely.geometry.Polygon(cells[i][j])
                    if calc_area:
                        if pol.intersects(cell):
                            res[i, j] = pol.intersection(cell).area
                    else:
                        inters = pol.intersects(cell)
                        if inters:
                            res[i, j] = 1
            return res
        elif len(index) == 3:
            i1 = index[0]
            i2 = index[1]
            i3 = index[2]
            for i in range(0, 3):
                assert mesh.axes[i].dimension == 1
            try:
                cells = mesh.axes[i1].cell_edges3d(mesh.axes[i2], mesh.axes[i3])
            except (TypeError, AttributeError) as e:
                raise type(e)(e.message + "Custom axis should implement cell_edges3d method. "
                                          "This method returns 3d list of ordered sequence of point tuples."
                                          " See docstring for more information.")
            shape = (mesh.axes[i1].size, mesh.axes[i2].size, mesh.axes[i3].size)
            res = np.zeros(shape)
            for k, plane in enumerate(res):
                for i, row in enumerate(plane):
                    for j, _ in enumerate(row):
                        cell = shapely.geometry.Polygon(cells[k][i][j])
                        if calc_area:
                            if pol.intersects(cell):
                                res[k, i, j] = pol.intersection(cell).area
                        else:
                            inters = pol.intersects(cell)
                            if inters:
                                res[k, i, j] = 1
            return res
        else:
            raise TypeError("Unknown axes formation.")
