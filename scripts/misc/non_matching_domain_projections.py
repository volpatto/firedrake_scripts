from firedrake import *

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# Global mesh attributes
use_quads = False  # not working for quads

# Fine mesh
num_elements_x_fine = 10
num_elements_y_fine = 10
use_quads_fine_mesh = use_quads
fine_mesh = UnitSquareMesh(
    num_elements_x_fine, num_elements_y_fine, quadrilateral=use_quads_fine_mesh
)

# Coarse mesh
num_elements_x_coarse = 200
num_elements_y_coarse = 200
size = 10.0
use_quads_coarse_mesh = use_quads
coarse_mesh = SquareMesh(
    num_elements_x_coarse, num_elements_y_coarse, L=size, quadrilateral=use_quads_coarse_mesh
)

# Coarse and fine function spaces
coarse_degree = 4
coarse_family = "DQ" if use_quads_coarse_mesh else "DG"
V_coarse = FunctionSpace(coarse_mesh, coarse_family, coarse_degree)

fine_degree = 1
fine_family = "DQ" if use_quads_fine_mesh else "DG"
V_fine = FunctionSpace(fine_mesh, fine_family, fine_degree)

# A function on coarse mesh
value_coarse = Constant(0)
u_coarse = project(value_coarse, V_coarse)

# A function on fine mesh
value_fine = Constant(1)
u_fine = project(value_fine, V_fine)

# Projecting u_fine in u_coarse
u_composed = project(u_fine, V_coarse)

# Plotting projected result
fig, axes = plt.subplots()
collection = tripcolor(u_composed, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("projection_fine_in_coarse.png")
