from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# Mesh definition
numel_x = 10
numel_y = 10
Lx, Ly = 1.0, 1.0
mesh = RectangleMesh(numel_x, numel_y, Lx, Ly, quadrilateral=True)

# Defining the Function Spaces
degree = 1
V = FunctionSpace(mesh, "CG", degree)

# Trial and Test functions declaration
u = TrialFunction(V)
v = TestFunction(V)

# Declaring the Boundary Conditions
u_boundary = 0.0
g_bound = Constant(u_boundary)

# Marking the BC
bcs = DirichletBC(V, g_bound, 'on_boundary')

# Source term
f = interpolate(Constant(1.0), V)

# Model parameters
k = Constant(1e-8)
c = Constant(1.0)

# Bilinear form
a = k * inner(grad(v), grad(u)) * dx + c * inner(u, v) * dx

# Linear form (RHS)
L = f * v * dx

# *** Stabilizing terms ***
# Stabilizing parameters (see Franca and Do Carmo (1988))
h = CellDiameter(mesh)
alpha = (c * h * h)/(6.0 * k)
eps = conditional(ge(alpha, 8), 1, conditional(ge(alpha, 1), 0.064 * alpha + 0.49, 0))
tau = (eps * (h * h))/(6.0 * c)
# Adding GGLS stabilizing terms
a += inner(grad(c * u - k * div(grad(u))), tau * grad(c * v - k * div(grad(v)))) * dx
L += inner(grad(f), tau * grad(c * v - k * div(grad(v)))) * dx

# Mounting the discrete variational problem
u_sol = Function(V)
problem = LinearVariationalProblem(a, L, u_sol, bcs)

# Solving the problem
solver = LinearVariationalSolver(problem)
solver.solve()

# Displaying
fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
collection = trisurf(u_sol, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("solution_pressure.png")

# Writing the solution in pvd/vtu
# outfile = File("../outputs/ggls2d.pvd")
# outfile.write(u_sol)
