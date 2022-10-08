from firedrake import *
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text')
plt.rc('font', size=14)

Lx, Ly = 1.0, 0.1
numel_x, numel_y = 15, 10
mesh = RectangleMesh(numel_x, numel_y, Lx, Ly, quadrilateral=True)

# Defining the vector function space to primal solution of linear elasticity problem
V = VectorFunctionSpace(mesh, "CG", 1)

# Declaring the trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Linear elasticity parameters
rho = Constant(0.01)
g = Constant(9.8)
f = as_vector([0, -rho * g])
mu = Constant(1)
lambda_ = Constant(0.45)
Id = Identity(mesh.geometric_dimension())


# Strain and Stress definitions
def epsilon(u):
    return (1. / 2.) * (grad(u) + grad(u).T)


def sigma(u):
    return lambda_ * div(u) * Id + 2 * mu * epsilon(u)


# Variational formulation
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx

# Boundary conditions
ux_bc, uy_bc = 0.0, 0.0
bcs = DirichletBC(V, Constant([ux_bc, uy_bc]), [1, 2])  # Dirichlet conditions at left and right (marks 1 and 2) boundary

# Solving the problem
u_h = Function(V)
solve(a == L, u_h, bcs=bcs)

# Stiffness matrix assembling
A = assemble(a, bcs=bcs)

# Printing the stiffness matrix entries and plotting
A_entries = A.M.values
plt.spy(A_entries)
plt.show()

# *** Plotting the displacement ***

# Creating the figure object
fig, axes = plt.subplots(figsize=(20, 2))

# Plotting the solution
collection = quiver(u_h, axes=axes)
fig.colorbar(collection)
axes.set_aspect("equal")

# Setting the xy-labels
plt.xlabel('x [L]')
plt.ylabel('y [L]')

# Saving the figure
plt.savefig("solution_displacement_quiver.png")

# Displaying in the screen
plt.show()

# *** Plotting x-axis displacements ***

fig, axes = plt.subplots(figsize=(10, 2))

# Componente x
collection = tripcolor(u_h.sub(0), axes=axes)
fig.colorbar(collection)
axes.set_aspect("equal")

plt.xlabel('x [L]')
plt.ylabel('y [L]')

axes.set_xlim([0, 1])

# plt.show()
plt.savefig("solution_displacement_x.png")

# *** Plotting y-axis displacements ***

fig, axes = plt.subplots(figsize=(10, 2))

# Componente x
collection = tripcolor(u_h.sub(1), axes=axes)
fig.colorbar(collection)
axes.set_aspect("equal")

plt.xlabel('x [L]')
plt.ylabel('y [L]')

axes.set_xlim([0, 1])

# plt.show()
plt.savefig("solution_displacement_y.png")

# *** Plotting the bar deformation ***

# Creating the deformed mesh
displaced_coordinates = interpolate(SpatialCoordinate(mesh) + u_h, V)
displaced_mesh = Mesh(displaced_coordinates)

# Creating the figure object
fig, axes = plt.subplots(figsize=(8, 6))

# Plotting the deformed mesh
triplot(displaced_mesh, axes=axes)
axes.set_aspect("equal")

# Setting the xy-labels
plt.xlabel(r'$x$ [L]')
plt.ylabel(r'$y$ [L]')

# Displaying in the screen
plt.tight_layout()
plt.savefig("displaced_mesh.png")
# plt.show()
