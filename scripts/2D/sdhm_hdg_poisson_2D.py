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
N = 10
use_quads = False
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads)
comm = mesh.comm

# Function space declaration
pressure_family = 'DQ' if use_quads else 'DG'
velocity_family = 'DQ' if use_quads else 'DG'
degree = 1
is_multiplier_continuous = False
U = VectorFunctionSpace(mesh, velocity_family, degree)
V = FunctionSpace(mesh, pressure_family, degree)
if is_multiplier_continuous:
    LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
    C0TraceElement = LagrangeElement["facet"]
    T = FunctionSpace(mesh, C0TraceElement)
else:
    trace_family = "DGT"
    T = FunctionSpace(mesh, trace_family, degree)
W = U * V * T

# Trial and test functions
solution = Function(W)
u, p, lambda_h = split(solution)
# u, p, lambda_h = TrialFunctions(W)
v, q, mu_h  = TestFunctions(W)

# Mesh entities
n = FacetNormal(mesh)
h = CellDiameter(mesh)
x, y = SpatialCoordinate(mesh)

# Exact solution
p_exact = sin(2 * pi * x) * sin(2 * pi * y)
# p_exact = sin(0.5 * pi * x) * sin(0.5 * pi * y)
exact_solution = Function(V).interpolate(p_exact)
exact_solution.rename("Exact pressure", "label")
sigma_e = Function(U, name='Exact velocity')
sigma_e.project(-grad(p_exact))

# Forcing function
f_expression = div(-grad(p_exact))
f = Function(V).interpolate(f_expression)

# Dirichlet BCs
bc_multiplier = DirichletBC(W.sub(2), p_exact, "on_boundary")

# Hybridization parameter
beta_0 = Constant(1.0e0)
# beta = beta_0 / h
beta = beta_0

# Least-Squares parameters
delta_1 = Constant(-0.0) * h * h
delta_2 = Constant(0.0) * h * h
delta_3 = Constant(0.0) * h * h

# Numerical flux trace
u_hat = u + beta * (p - lambda_h) * n

# HDG classical form
a = (dot(u, v) - div(v) * p) * dx + lambda_h("+") * jump(v, n) * dS
a += -dot(u, grad(q)) * dx + jump(u_hat, n) * q("+") * dS
L = f * q * dx

# Least-squares terms
a += delta_1 * inner(u + grad(p), v + grad(q)) * dx
a += delta_2 * div(u) * div(v) * dx
a += delta_3 * inner(curl(u), curl(v)) * dx
L += delta_2 * f * div(v) * dx

# Transmission condition
a += jump(u_hat, n) * mu_h("+") * dS

# Weakly imposed BC
# a += lambda_h * dot(v, n) * ds  # required term
L += -exact_solution * dot(v, n) * ds  # required as the above, but just one of them should be used
a += dot(u, n) * q * ds + beta * p * q * ds  # required term... note that u (the unknown) is used
L += beta * exact_solution * q * ds  # Required, this one is paired with the above term
a += lambda_h * mu_h * ds  # Classical required term
L += exact_solution * mu_h * ds  # Pair for the above classical required term

F = a - L

# Solving with Static Condensation
print("*******************************************\nSolving using static condensation.\n")
params = {
    "snes_type": "ksponly",
    "mat_type": "matfree",
    "pmat_type": "matfree",
    "ksp_type": "preonly",
    "pc_type": "python",
    # Use the static condensation PC for hybridized problems
    # and use a direct solve on the reduced system for lambda_h
    "pc_python_type": "firedrake.SCPC",
    "pc_sc_eliminate_fields": "0, 1",
    "condensed_field": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_monitor_true_residual": None,
    },
}

problem = NonlinearVariationalProblem(F, solution, bcs=[])
# problem = NonlinearVariationalProblem(F, solution)
solver = NonlinearVariationalSolver(problem, solver_parameters=params)
solver.snes.ksp.setConvergenceHistory()
solver.solve()
print("Solver finished.\n")

# Solving fully coupled system
# print("*******************************************\nSolving fully coupled system.\n")
# solver_parameters = {
#     "ksp_monitor": None,
#     "mat_type": "aij",
#     "ksp_type": "preonly",
#     "pc_type": "lu",
#     "pc_factor_mat_solver_type": "mumps",
# }
# solution = Function(W)
# problem = LinearVariationalProblem(a, L, solution, bcs=[])
# solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
# solver.snes.ksp.setConvergenceHistory()
# solver.solve()
# print("Solver finished.\n")

sigma_h, u_h, lambda_h = solution.split()
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
