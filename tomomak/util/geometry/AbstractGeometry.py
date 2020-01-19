
class AbstractGeometry:
    """Routines to work with geometry
    """

    @staticmethod
    def intersection(mesh, points, index=(0, 1), calc_area=True):
        """Create solution array, representing polygon, defined by specified points on the given mesh.

        If there are more than N dimension in model, broadcasting to other dimensions is performed.
        If broadcasting is not needed private method _polygon may be used.

        Args:
            mesh(tomomak.main_structures.Mesh): mesh to work with.
            points(An ordered sequence of point tuples, optional): Polygon points (x, y, z ...).
            index(tuple of ints, optional): axes to build object at.
            calc_area(bool): If True, area of intersection with each cell is calculated, if False,
                only fact of intersecting with mesh cell is taken into account. Default: True.

        Returns:
            ndarray: numpy array, representing polygon on the given mesh.

        Raises:
            TypeError if one of the axes is not  cartesian (tomomak.main_structures.mesh.cartesian).
        """

    @staticmethod
    def cell_areas(mesh, index):
        """Get area of each cell on mesh.

        Args:
            mesh(tomomak.main_structures.Mesh): mesh to work with.
            index(tuple of ints, optional): axes to build object at.

        Returns:
            ndarray: ndarray with cell areas.

        """

    @staticmethod
    def cell_distances(mesh, index, p):
        """Get distance to each cell on mesh.

        Args:
            mesh(tomomak.main_structures.Mesh): mesh to work with.
            index(tuple ints, optional): axes to calculate distance at.

        Returns:
            ndarray: ndarray with distances.
            """

    @staticmethod
    def line_to_polygon(p1, p2, width, divergence=0):
        """Generate detector geometry for one Line of Sight.

        line of sight can be collimated or diverging (cone-like).

        Args:
            p1, p2(tuple of points):
            width(float): line width.
            divergence(float, optional): Angle between two LOS borders in Rad [0, pi). 0 means that line is collimated.
                default: 0.

        Returns:
            tuple of points: List of Shapely points, defining LOS.

        Raises:
            ValueError if divergence is < 0 or >= pi.
        """
