from tomomak import model
from tomomak.mesh import mesh
from tomomak.mesh import spiderweb_axes
from tomomak.test_objects import objects2d
from tomomak.util.gfileextractor import gfile_extract

border, center = gfile_extract('data/g035685.00150')
spider_axis = spiderweb_axes.SpiderWeb2dAxis(border=border, center=center)
mesh = mesh.Mesh((spider_axis,))

# Now we can create Model.
# Model is one of the basic tomomak structures which stores information about geometry, solution and detectors.
# At present we only have information about the geometry.
mod = model.Model(mesh=mesh)
# Now let's create synthetic 2D object to study.
# We will consider triangle.
real_solution = objects2d.polygon(mesh, [(0.2, -0.2), (0.5, -0.1), (0.30, 0.30)])
# Model.solution is the solution we are looking for.
# It will be obtained at the end of this example.
# However, if you already know supposed solution (for example you get it with simulation),
# you can use it as first approximation by setting Model.solution = *supposed solution*.
# Recently we've generated test object, which is, of course, real solution.
# A trick to visualize this object is to temporarily use it as model solution.
mod.solution = real_solution
mod.plot2d()
