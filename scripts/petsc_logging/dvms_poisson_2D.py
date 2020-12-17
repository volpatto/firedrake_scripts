from firedrake import *
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc


def print(content_to_print):
    return PETSc.Sys.Print(content_to_print)


parameters["pyop2_options"]["lazy_evaluation"] = False
PETSc.Log.begin()

# Defining the mesh
N = 10
use_quads = True
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads)
comm = mesh.comm

# Function space declaration
pressure_family = 'DQ' if use_quads else 'DG'
velocity_family = 'DQ' if use_quads else 'DG'
degree = 1
U = VectorFunctionSpace(mesh, velocity_family, degree)
V = FunctionSpace(mesh, pressure_family, degree)
W = U * V

# Trial and test functions
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

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

# Exact solution
p_exact = sin(2 * pi * x) * sin(2 * pi * y)
exact_solution = Function(V).interpolate(p_exact)
exact_solution.rename("Exact pressure", "label")
sigma_e = Function(U, name='Exact velocity')
sigma_e.project(-grad(p_exact))

# Forcing function
f_expression = div(-grad(p_exact))
f = Function(V).interpolate(f_expression)

# Dirichlet BCs
bcs = DirichletBC(W[0], sigma_e, "on_boundary", method="geometric")

# # Variational form
# a = inner(grad(u), grad(v)) * dx
# L = f * v * dx

# Average cell size and mesh dependent stabilization
h_avg = (h("+") + h("-")) / 2.0

# Jump stabilizing parameters based on Badia-Codina stabilized dG method
L0 = 1
# eta_p = L0 * h_avg  # method B in the Badia-Codina paper
eta_p = 1
# eta_p = L0 * L0  # method D in the Badia-Codina paper
# eta_u = h_avg / L0  # method B in the Badia-Codina paper
eta_u = 1
# eta_u_bc = h / L0  # method B in the Badia-Codina paper
eta_u_bc = 1

# Mixed classical terms
a = (dot(u, v) - div(v) * p + q * div(u)) * dx
L = f * q * dx
# DG terms
a += jump(v, n) * avg(p) * dS - avg(q) * jump(u, n) * dS
# Edge stabilizing terms
# ** Badia-Codina based
# a += (eta_p / h_avg) * (jump(u, n) * jump(v, n)) * dS
# a += (eta_u / h_avg) * dot(jump(p, n), jump(q, n)) * dS
# ** Mesh independent (original)
a += jump(u, n) * jump(v, n) * dS  # not considered in the original paper
a += dot(jump(p, n), jump(q, n)) * dS
# Volumetric stabilizing terms
a += 0.5 * inner(u + grad(p), grad(q) - v) * dx
a += 0.5 * h * h * div(u) * div(v) * dx
# a += 0.5 * h * h * inner(curl(u), curl(v)) * dx
# a += 0.5 * div(u) * div(v) * dx
# a += 0.5 * inner(curl(u), curl(v)) * dx
L += 0.5 * h * h * f * div(v) * dx

# Solving the system
solver_parameters = {
    'ksp_monitor': None,
    'ksp_view': None,
    'ksp_type': 'gmres',
    'pc_type': 'ilu',
    'mat_type': 'aij',
    'ksp_rtol': 1e-8,
    'ksp_max_it': 2000
}
solution = Function(W)
problem = LinearVariationalProblem(a, L, solution, bcs=bcs)
solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.snes.ksp.setConvergenceHistory()
solver.solve()

sigma_h, u_h = solution.split()
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

# Plotting velocity field exact solution
fig, axes = plt.subplots()
collection = quiver(sigma_e, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exact solution for velocity")
plt.show()

# Plotting pressure field exact solution
fig, axes = plt.subplots()
collection = tripcolor(exact_solution, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exact solution for pressure")
plt.show()

# Plotting velocity field numerical solution
fig, axes = plt.subplots()
collection = quiver(sigma_h, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plotting pressure field numerical solution
fig, axes = plt.subplots()
collection = tripcolor(u_h, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
