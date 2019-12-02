"""Generators for basic detectors arrays in 2D geometry
"""
import shapely.geometry
import shapely.affinity
import tomomak.util.geometry2d
import numpy as np


def line2d(mesh, p1, p2, width, divergence, index=(0, 1), response=1, radius_dependence=True,
           broadcast=True, calc_area=True):
    """
        Source is isotropic.
    Args:
        mesh:
        p1(tuple of 2 floats): Detector origin (x, y).
        p2:
        width:
        divergence:
        index:
        response(float): Detector response = amplification * detector area.
            E.g. detector signal at 1m from source, emitting 4*pi particles at given time interval.
        radius_dependence:
        broadcast:
        calc_area: =True

    Returns:

    """
    points = _line_to_polygon(p1, p2, width, divergence)
    if isinstance(index, int):
        index = [index]
    # Get intersection area
    res = tomomak.util.geometry2d.intersection_2d(mesh, points, index, calc_area)

    if radius_dependence:
        r = np.ones_like(res)
        p1 = shapely.geometry.Point(p1)
        # If axis is 2D
        if mesh.axes[index[0]].dimension == 2:
            i1 = index[0]
            for i, row in enumerate(res):
                if res[i]:
                    p2 = shapely.geometry.Point(mesh.axes[i1].coordinates[i])
                    r[i] = p1.distance(p2)
        # If axes are 1D
        elif mesh.axes[0].dimension == 1:
            i1 = index[0]
            i2 = index[1]
            for i, row in enumerate(res):
                for j, _ in enumerate(row):
                    if res[i, j]:
                        p2 = shapely.geometry.Point(mesh.axes[i1].coordinates[i], mesh.axes[i2].coordinates[j])
                        r[i, j] = p1.distance(p2)
        r = 4 * np.pi * np.square(r)
        res /= r
    res *= response
    if broadcast:
        res = tomomak.util.array_routines.broadcast_object(res, index, mesh.shape)
    return res


# def fan_detector(mesh, position, number, width, incline, angle=np.pi, divergence=0,
#                  index=(0, 1), response=1, radius_dependence=True, broadcast=True):
#     """
#
#     Args:
#         mesh:
#         position:
#         number:
#         width:
#         incline:
#         angle:
#         divergence:
#         index:
#         response:
#         radius_dependence:
#         broadcast:
#
#     Returns:
#
#     """
#     """
#     creates one fan of detectors
#     line_num - number of sightlines
#     N,M - grid size
#     (note, that pixels are rectangular, i.e. the  ratio of the investigated area in the real world is N/M)
#     N corresponds to x, M - to y, note that physycal coordinates and NumPy array coordinates are different
#     width - width of one sightline
#     angle - fan angle
#     position - position of the fan vertex in radians
#     (counter-clockwise, 0 is middle-right as cos-sin representation)
#     shift - shift of the fan vertex in pixels
#     incline - incline of the fan from the center in radians
#     weighted - if False, only the fact of intersection of the sightline and pixel is considered
#     if True - area of the sightline and pixel intersection is considered
#     r2dep - if dependence of sygnal intensity on r^2 exists, warning: normalization of this parameter is pixel length = 1
#     """
#     #create array of polygons, representing grid
#     box_ar = np.empty([M,N],dtype=Polygon)
#     for j in range (M):
#             for k in range (N):
#                 box_ar[j,k] = box(k,j,k+1,j+1)
#     #finding first sightline of the detector
#     r = sqrt(M * M  + N * N ) / 2 + shift
#     p1 = (N/2, M/2)
#     p2 = (N/2 + r + 1,M/2)
#     l = LineString([p1,p2])
#     l = rotate(l, -pos, origin = p1, use_radians=True)
#     grid_box = LinearRing([(0,0),(N,0),(N,M),(0,M)])
#     p3 = grid_box.intersection(l)
#     l = LineString ([p1,p3])
#     sc_factor = (l.length + shift) / l.length
#     l = scale (l, xfact = sc_factor, yfact = sc_factor, origin = p1)
#     ptop = l.coords[1]
#     l = scale (l,xfact = 2, yfact = 2, origin = ptop)
#     l = rotate(l, -angle/2, origin = ptop, use_radians=True)
#     l = rotate(l, incline, origin = ptop, use_radians=True)
#
#     w = np.zeros((number, M, N))      #basis for the resulting array of weights
#     d_angle = angle / number          #angle between two sightlines
#
#     #start scanning
#     for i in range (number):
#         det_line = line_to_det(l,width,divergence ) #sightline polygon which considers finite line width
#
#         if weighted:
#             for j in range (M):
#                 for k in range (N):
#                     if det_line.intersects(box_ar[j,k]):
#                         w[i,j,k] = det_line.intersection(box_ar[j,k]).area
#         else:
#             for j in range (M):
#                 for k in range (N):
#                     w[i,j,k] = det_line.intersects(box_ar[j,k])
#         if r2dep:
#             for j in range (M):
#                 for k in range (N):
#                     x = j - ptop[1]
#                     y = k - ptop[0]
#                     r2 = x**2+y**2
#                     w[i,j,k] = w[i,j,k] / r2
#
#
#
#         l = rotate(l,d_angle,origin=ptop,use_radians=True) #rotate sightline for the next step
#     return  w

def _line_to_polygon(p1, p2, width, divergence=0):
    """Generate detector geometry for one Line of Sight.

    line of sight can be collimated or diverging (cone-like).

    Args:
        p1, p2(tuple of 2 points):
        width(float): line width.
        divergence(float, optional): Angle between two LOS borders in Rad [0, pi). 0 means that line is collimated.
            default: 0.

    Returns:
        tuple of 4 points: List of 4 Shapely points, defining LOS.

    Raises:
        ValueError if divergence is < 0 or >= pi.
    """
    if divergence < 0 or divergence > np.pi:
        raise ValueError("Divergence value is {}. It should be >= 0 and < pi.".format(divergence))
    line = shapely.geometry.LineString([p1, p2])
    ll = line.parallel_offset(width/2, 'left')
    p_top = ll.coords[0]
    ll = shapely.affinity.rotate(ll, divergence/2, origin=p_top, use_radians=True)
    lr = line.parallel_offset(width/2, 'right')
    p_top = lr.coords[1]
    lr = shapely.affinity.rotate(lr, -divergence/2, origin=p_top, use_radians=True)
    p1, p2 = ll.coords
    p3, p4 = lr.coords
    return p1, p2, p3, p4
