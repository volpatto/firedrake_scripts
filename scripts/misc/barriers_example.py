from firedrake import *
import matplotlib.pyplot as plt

elements_in_x = 20
elements_in_y = 20
size_x = 100.0
size_y = 50.0
use_quads = True
mesh = RectangleMesh(elements_in_x, elements_in_y, size_x, size_y, quadrilateral=use_quads)
V = FunctionSpace(mesh, "DG", 1)
x, y = SpatialCoordinate(mesh)

obstacle_value = 1.0
free_path_value = 0.0
K = conditional(
    Or(
        And(
            And(ge(x, 20.0), le(x, 40.0)) , And(ge(y, 0.0), le(y, 25.0))
        ), 
        And(
            And(ge(x, 60.0), le(x, 80.0)), And(ge(y, 25.0),le(y, 50.0))
        )
    ), 
    Constant(obstacle_value), 
    Constant(free_path_value)
)

# The correct approach to have the right values on the nodes is "project", not "interpolate"
K_in_mesh = project(K, V)

fig, axes = plt.subplots()
collection = tripcolor(K_in_mesh, axes=axes, cmap='coolwarm', shading='flat')
fig.colorbar(collection)
triplot(mesh, axes=axes)
axes.set_xlim([0, size_x])
axes.set_ylim([0, size_y])
plt.xlabel("x")
plt.ylabel("y")
plt.title("K field")
plt.savefig("K_barriers.png")
