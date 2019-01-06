from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', size=14)

# Defining the mesh in the interval line
x_left = 0.0
x_right = 1.0
numel = 20
mesh = IntervalMesh(numel, x_left, x_right)
mesh_ref = IntervalMesh(200, x_left, x_right)

# Defining the Function Spaces
p = 1
V = FunctionSpace(mesh, "CG", p)
Vref = FunctionSpace(mesh_ref, "CG", p)

# Trial and Test functions declaration
u = TrialFunction(V)
v = TestFunction(V)

# Declaring the Boundary Conditions
u1, u2 = 0.0, 0.0
g_left = Constant(u1)
g_right = Constant(u2)

# Marking the BC
bc_left = DirichletBC(V, g_left, 1)
bc_right = DirichletBC(V, g_right, 2)
dirichlet_condition = [bc_left, bc_right]

# Source term
f = interpolate(Expression(1.0), V)

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
problem = LinearVariationalProblem(a, L, u_sol, dirichlet_condition)

# Solving the problem
solver = LinearVariationalSolver(problem)
solver.solve()

# Exact solution
sol_exact = Expression('x[0]<=0+tol || x[0]>=1-tol ? 0 : 1', degree=1, tol=1e-10)
u_e = interpolate(sol_exact, Vref)

# *** Plotting ***

# Getting the data to numpy array
x_values = mesh.coordinates.vector().dat.data
x_ref = mesh_ref.coordinates.vector().dat.data
sol_vec = u_sol.vector().dat.data
exact_vec = u_e.vector().dat.data

# Creating the figure instance
fig = plt.figure(dpi=300, figsize=(8, 6))
ax = plt.subplot(111)

# Plotting the data
ax.plot(x_values, sol_vec, '-x', label='GGLS')
ax.plot(x_ref, exact_vec, 'k-', label='Exact')

# Getting and setting the legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, 1.01 * box.width, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Setting the xy-labels
plt.xlabel(r'$x$')
plt.ylabel(r'$u$')
plt.xlim(x_values.min(), x_values.max())
plt.ylim(sol_vec.min(), 1.02 * sol_vec.max())

# Setting the grids in the figure
plt.minorticks_on()
plt.grid(True)
plt.grid(False, linestyle='--', linewidth=0.5, which='major')
plt.grid(False, linestyle='--', linewidth=0.1, which='minor')

# Displaying the plot
plt.show()
