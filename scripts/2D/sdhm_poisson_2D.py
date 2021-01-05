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
N = 20
use_quads = True
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads)
comm = mesh.comm

# Function space declaration
pressure_family = 'DQ' if use_quads else 'DG'
velocity_family = 'DQ' if use_quads else 'DG'
trace_family = "HDiv Trace"
degree = 1
U = VectorFunctionSpace(mesh, velocity_family, degree)
V = FunctionSpace(mesh, pressure_family, degree)
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
exact_solution = Function(V).interpolate(p_exact)
exact_solution.rename("Exact pressure", "label")
sigma_e = Function(U, name='Exact velocity')
sigma_e.project(-grad(p_exact))

# Forcing function
f_expression = div(-grad(p_exact))
f = Function(V).interpolate(f_expression)

# Dirichlet BCs
bcs = DirichletBC(W[0], sigma_e, "on_boundary", method="geometric")
bc_multiplier = DirichletBC(W.sub(2), Constant(0.0), "on_boundary")

# BCs
p_boundaries = Constant(0.0)
u_projected = sigma_e

# Hybridization parameter
beta_0 = Constant(1.0e-18)
# beta = beta_0 / h
beta = beta_0
beta_avg = beta_0 / h("+")

# Mixed classical terms
a = (dot(u, v) - div(v) * p - q * div(u)) * dx
L = -f * q * dx
# Stabilizing terms
a += -0.5 * inner(u + grad(p), v + grad(q)) * dx
a += 0.5 * div(u) * div(v) * dx
a += 0.5 * inner(curl(u), curl(v)) * dx
L += 0.5 * f * div(v) * dx
# Hybridization terms
a += lambda_h("+") * dot(v, n)("+") * dS + mu_h("+") * dot(u, n)("+") * dS
a += beta_avg * (lambda_h("+") - p("+")) * (mu_h("+") - q("+")) * dS
# Weakly imposed BC
a += (p_boundaries * dot(v, n) + mu_h * (dot(u, n) - dot(u_projected, n))) * ds
a += beta * (lambda_h - p_boundaries) * mu_h * ds

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
    "pc_sc_eliminate_fields": "0, 1",
    "condensed_field": {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_monitor_true_residual": None,
    },
}

# problem = NonlinearVariationalProblem(F, solution, bcs=bc_multiplier)
problem = NonlinearVariationalProblem(F, solution)
solver = NonlinearVariationalSolver(problem, solver_parameters=params)
solver.snes.ksp.setConvergenceHistory()
solver.solve()
PETSc.Sys.Print("Solver finished.\n")

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
