from firedrake import *
import matplotlib
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc

matplotlib.use('Agg')


def print(content_to_print):
    return PETSc.Sys.Print(content_to_print)


parameters["pyop2_options"]["lazy_evaluation"] = False

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
U = VectorFunctionSpace(mesh, velocity_family, degree - 1)  # just for post-processing
V = FunctionSpace(mesh, pressure_family, degree)
V_exact = FunctionSpace(mesh, pressure_family, degree + 3)
if is_multiplier_continuous:
    LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
    C0TraceElement = LagrangeElement["facet"]
    T = FunctionSpace(mesh, C0TraceElement)
else:
    trace_family = "DGT"
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
# p_exact = sin(0.5 * pi * x) * sin(0.5 * pi * y)
exact_solution = Function(V).interpolate(p_exact)
exact_solution.rename("Exact pressure", "label")
sigma_e = Function(U, name='Exact velocity')
sigma_e.interpolate(-grad(p_exact))

# Forcing function
f_expression = div(-grad(p_exact))
f = Function(V_exact).interpolate(f_expression)

# Dirichlet BCs
bc_multiplier = DirichletBC(W.sub(1), p_exact, "on_boundary")

# Hybridization parameter
beta_0 = Constant(8 * degree * degree)  # rather arbitrary
beta = beta_0 / h
# beta = beta_0

# Numerical flux trace
u = -grad(p)
u_hat = u + beta * (p - lambda_h) * n

# Symmetry parameter: s = -1 (symmetric), s = 1 (unsymmetric), and s = 0 (no-symmetrization)
s = Constant(-1)

# Classical Galerkin form
a = -dot(u, grad(q)) * dx + jump(u_hat, n) * q("+") * dS
# a = div(u) * q * dx
L = f * q * dx
# Transmission condition
a += -jump(u_hat, n) * mu_h("+") * dS
# Symmetrization
a += s * jump(grad(q), n) * (p('+') - lambda_h("+")) * dS
a += s * dot(grad(q), n) * (p - exact_solution) * ds
# Weakly imposed BC
# a += dot(u_hat, n) * q * ds
a += dot(u, n) * q * ds	+ beta * (p - exact_solution) * q * ds  # expand u_hat product in ds
a += mu_h * (lambda_h - exact_solution) * ds

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
    "pc_sc_eliminate_fields": "0",
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
solver.solve()
print("Solver finished.\n")

u_h, lambda_h = solution.split()
sigma_h = Function(U, name='Velocity')
sigma_h.interpolate(-grad(u_h))
u_h.rename('Pressure', 'label')

# Plotting velocity field exact solution
fig, axes = plt.subplots()
# triplot(mesh, axes=axes)
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
triplot(mesh, axes=axes)
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exact solution for pressure")
plt.savefig("exact_pressure.png")
# plt.show()

# Plotting velocity field numerical solution
fig, axes = plt.subplots()
# triplot(mesh, axes=axes)
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
triplot(mesh, axes=axes)
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("solution_pressure.png")
# plt.show()
