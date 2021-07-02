from firedrake import *
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc


def print(content_to_print):
    return PETSc.Sys.Print(content_to_print)


parameters["pyop2_options"]["lazy_evaluation"] = False

# Defining the mesh
N = 20
use_quads = True
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads)
comm = mesh.comm

# Function space declaration
pressure_family = 'DQ' if use_quads else 'DG'
velocity_family = 'DQ' if use_quads else 'DG'
degree = 1
U = VectorFunctionSpace(mesh, velocity_family, degree - 1)  # just for post-processing
V = FunctionSpace(mesh, pressure_family, degree)

# Trial and test functions
p = TrialFunction(V)
q = TestFunction(V)

# Mesh entities
n = FacetNormal(mesh)
h = CellDiameter(mesh)
x, y = SpatialCoordinate(mesh)

# Exact solution
p_exact = sin(2 * pi * x) * sin(2 * pi * y)
exact_solution = Function(V).interpolate(p_exact)
exact_solution.rename("Exact pressure", "label")
sigma_e = Function(U, name='Exact velocity')
sigma_e.project(-grad(p_exact))

# Forcing function
f_expression = div(-grad(p_exact))
f = Function(V).interpolate(f_expression)

# Edge stabilizing parameter
beta0 = Constant(1e1)
beta = beta0 / h

# Symmetry term. Choose if the method is SIPG (-1) or NIPG (1)
s = Constant(-1)

# Classical volumetric terms
a = inner(grad(p), grad(q)) * dx
L = f * q * dx
# DG edge terms
a += s * dot(jump(p, n), avg(grad(q))) * dS - dot(avg(grad(p)), jump(q, n)) * dS
# Edge stabilizing terms
a += beta("+") * dot(jump(p, n), jump(q, n)) * dS
# Weak boundary conditions
a += s * dot(p * n, grad(q)) * ds - dot(grad(p), q * n) * ds
a += beta * p * q * ds
L += beta * exact_solution * q * ds

# Solving the system
solver_parameters = {
    "ksp_monitor": None,
    "ksp_view": None,
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
solution = Function(V)
problem = LinearVariationalProblem(a, L, solution, bcs=[])
solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.solve()

u_h = solution
sigma_h = Function(U, name='Velocity')
sigma_h.project(-grad(u_h))
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

# Plotting velocity field exact solution
fig, axes = plt.subplots()
collection = quiver(sigma_e, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exact solution for velocity")
plt.savefig("exact_velocity.png")
# plt.show()

# Plotting pressure field exact solution
fig, axes = plt.subplots()
collection = tripcolor(exact_solution, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exact solution for pressure")
plt.savefig("exact_pressure.png")
# plt.show()

# Plotting velocity field numerical solution
fig, axes = plt.subplots()
collection = quiver(sigma_h, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("solution_velocity.png")
# plt.show()

# Plotting pressure field numerical solution
fig, axes = plt.subplots()
collection = tripcolor(u_h, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("solution_pressure.png")
# plt.show()
