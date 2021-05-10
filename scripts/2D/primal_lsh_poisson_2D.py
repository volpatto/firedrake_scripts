from firedrake import *
import matplotlib
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc

matplotlib.use('Agg')


def print(content_to_print):
    return PETSc.Sys.Print(content_to_print)


parameters["pyop2_options"]["lazy_evaluation"] = False
PETSc.Log.begin()

# Defining the mesh
N = 15
use_quads = True
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads)
comm = mesh.comm

# Function space declaration
is_multiplier_continuous = True
pressure_family = 'DQ' if use_quads else 'DG'
velocity_family = 'DQ' if use_quads else 'DG'
degree = 1
U = VectorFunctionSpace(mesh, velocity_family, degree)
V = FunctionSpace(mesh, pressure_family, degree + 1)
if is_multiplier_continuous:
    LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
    C0TraceElement = LagrangeElement["facet"]
    T = FunctionSpace(mesh, C0TraceElement)
else:
    trace_family = "HDiv Trace"
    T = FunctionSpace(mesh, trace_family, degree)
W = V * T

# Trial and test functions
solution = Function(W)
p, lambda_h = split(solution)
# u, p, lambda_h = TrialFunctions(W)
q, mu_h  = TestFunctions(W)

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

# Dirichlet BCs
bc_multiplier = DirichletBC(W.sub(1), p_exact, "on_boundary")
# bcs = DirichletBC(W[0], sigma_e, "on_boundary", method="geometric")

# BCs
p_boundaries = Constant(0.0)
u_projected = sigma_e

# Hybridization parameter
beta_0 = Constant(-32e0)
beta = beta_0 / h

# Stabilizing parameter
# delta_0 = Constant(1)
# delta_1 = Constant(0)
# delta_2 = Constant(1)
# delta_3 = Constant(1)
# delta_4 = Constant(1)
# delta_5 = Constant(1)
delta_1 = h * h
delta_2 = h * h
delta_3 = h * h * Constant(1e0)

# Flux variables
u = -grad(p)
v = -grad(q)

# Numerical flux trace
u_hat = u + beta * (p - lambda_h) * n
v_hat = v + beta * (q - mu_h) * n

# # Flux least-squares
# a = (
#     (inner(u, v) - q * div(u) - p * div(v) + inner(grad(p), grad(q)))
#     * delta_1
#     * dx
# )
# a += delta_1 * jump(u_hat, n=n) * q("+") * dS
# a += delta_1 * dot(u_hat, n) * q * ds
# # a += delta_1 * dot(u, n) * q * ds
# # L = -delta_1 * dot(u_projected, n) * q * ds
# a += delta_1 * lambda_h("+") * jump(v, n=n) * dS
# a += delta_1 * lambda_h * dot(v, n) * ds
# # L = -delta_1 * p_boundaries * dot(v, n) * ds

# Flux Least-squares as in DG
# a = delta_0 * inner(u + grad(p), v + grad(q)) * dx

# Mass balance least-square
a = delta_1 * div(u) * div(v) * dx
L = delta_1 * f * div(v) * dx

# Irrotational least-squares
# a += delta_1 * inner(curl(u), curl(v)) * dx

# Hybridization terms
a += mu_h("+") * jump(u_hat, n=n) * dS
a += avg(delta_2) * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
a += delta_2 * (p - lambda_h) * (q - mu_h) * ds
a += avg(delta_3) * (dot(u, n)("+") - dot(u_hat, n)("+")) * (dot(v, n)("+") - dot(v_hat, n)("+")) * dS
a += delta_3 * (dot(u, n) - dot(u_hat, n)) * (dot(v, n) - dot(v_hat, n)) * ds
# a += delta_3 * (dot(u, n) - dot(u_projected, n)) * (dot(v, n) - dot(v_hat, n)) * ds
# a += delta_4 * (lambda_h("+") - p("+")) * (mu_h("+") - q("+")) * dS

F = a - L

# Solving with Static Condensation
PETSc.Sys.Print("*******************************************\nSolving using static condensation.\n")
params = {
    "snes_type": "ksponly",
    "mat_type": "matfree",
    "pmat_type": "matfree",
    "ksp_type": "preonly",
    "pc_type": "python",
    # Use the static condensation PC for hybridized problems
    # and use a direct solve on the reduced system for lambda_h
    "pc_python_type": "firedrake.SCPC",
    "pc_sc_eliminate_fields": "0",
    "condensed_field": {
        "ksp_type": "preonly",
        # 'ksp_view': None,
        "pc_type": "lu",
        "ksp_monitor": None,
        "pc_factor_mat_solver_type": "mumps",
        # "mat_mumps_icntl_4": "2",
        "ksp_monitor_true_residual": None
    },
}
# params = {
#     "snes_type": "ksponly",
#     "mat_type": "matfree",
#     "pmat_type": "matfree",
#     "ksp_type": "preonly",
#     "pc_type": "python",
#     # Use the static condensation PC for hybridized problems
#     # and use a direct solve on the reduced system for lambda_h
#     "pc_python_type": "firedrake.SCPC",
#     "pc_sc_eliminate_fields": "0, 1",
#     "condensed_field": {
#         "ksp_type": "preonly",
#         "pc_type": "svd",
#         'pc_svd_monitor': None,
#         'ksp_monitor_singular_value': None,
#         "pc_factor_mat_solver_type": "mumps",
#         'mat_type': 'aij'
#     },
# }

# problem = NonlinearVariationalProblem(F, solution)
problem = NonlinearVariationalProblem(F, solution, bcs=bc_multiplier)
solver = NonlinearVariationalSolver(problem, solver_parameters=params)
solver.snes.ksp.setConvergenceHistory()
solver.solve()
PETSc.Sys.Print("Solver finished.\n")

u_h, lambda_h = solution.split()
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
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
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
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("solution_pressure.png")
# plt.show()
