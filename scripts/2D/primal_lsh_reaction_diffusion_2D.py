from curses import KEY_BEG
from firedrake import *
import matplotlib
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc

matplotlib.use('Agg')


def print(content_to_print):
    return PETSc.Sys.Print(content_to_print)


# parameters["pyop2_options"]["lazy_evaluation"] = False
PETSc.Log.begin()

# Defining the mesh
N = 10
use_quads = False
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads, diagonal="left")
comm = mesh.comm

# Function space declaration
is_multiplier_continuous = False
pressure_family = 'DQ' if use_quads else 'DG'
velocity_family = 'DQ' if use_quads else 'DG'
degree = 1
U = VectorFunctionSpace(mesh, velocity_family, degree - 1)
V = FunctionSpace(mesh, pressure_family, degree)
V_exact = FunctionSpace(mesh, pressure_family, degree + 3)
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
p_exact = conditional(
    And(gt(x, 0), lt(x, 0.5)),
    conditional(And(gt(y, 0), lt(y, 1)), Constant(1), Constant(0)),
    Constant(0)
)
exact_solution = Function(V_exact).interpolate(p_exact)
exact_solution.rename("Exact pressure", "label")

# Forcing function
f_expression = p_exact
f = Function(V).project(f_expression)

# Dirichlet BCs
bc_multiplier = DirichletBC(W.sub(1), Constant(0), "on_boundary")

# Reaction and Diffusion coefficients
K = Constant(1e-8)
D = Constant(1)

# Hybridization parameter
beta_0 = Constant(K * degree * degree)  # should not be zero when used with LS terms
beta = beta_0 / h
# beta = beta_0

# GGLS stabilization parameters
alpha = (D * h * h) / (6.0 * K)
eps = conditional(ge(alpha, 8), 1, conditional(ge(alpha, 1), 0.064 * alpha + 0.49, 0))
tau = (eps * (h * h)) / (6.0 * D)

# Stabilizing parameter
# delta_base = h * h
delta_base = Constant(1)
delta_0 = delta_base * Constant(1)
delta_1 = delta_base * Constant(0)
# delta_2 = delta_base * Constant(1) / h
delta_2 = Constant(K * degree * degree) / h * Constant(0)
# delta_2 = beta
# delta_2 = delta_1 * Constant(1)  # so far this is the best combination
delta_3 = tau * delta_base * Constant(0)

# Flux variables
u = -K * grad(p)
v = -K * grad(q)

# Symmetry parameter: s = 1 (symmetric) or s = -1 (unsymmetric). Disable with 0.
s = Constant(-1)

# Numerical flux trace
u_hat = u + beta * (p - lambda_h) * n

# Classical term
a = delta_0 * dot(K * grad(p), grad(q)) * dx + delta_0 * D * p * q * dx
a += delta_0('+') * jump(u_hat, n) * q("+") * dS
# a += delta_0 * dot(u_hat, n) * q * ds
a += delta_0 * dot(u, n) * q * ds + delta_0 * beta * (p - exact_solution) * q * ds  # expand u_hat product in ds
L = delta_0 * f * q * dx

# Mass balance least-squares
a += delta_1 * (div(u) + D * p) * (div(v) + D * q) * dx
L += delta_1 * f * (div(v) + D * q) * dx

# Gradient mass balance least-squares
a += delta_3 * inner(grad(div(u) + D * p), grad(div(v) + D * q)) * dx
L += delta_3 * inner(grad(f), grad(div(v) + D * q)) * dx

# Hybridization terms
a += -mu_h("+") * jump(u_hat, n=n) * dS
a += mu_h * (lambda_h - exact_solution) * ds
# a += mu_h * dot(u_hat - grad(exact_solution), n) * ds

# Least-Squares on constrains
a += delta_2("+") * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
# a += delta_2 * (p - exact_solution) * (q - mu_h) * ds  # needed if not included as strong BC
a += delta_2 * (p - exact_solution) * q * ds  # needed if not included as strong BC
a += delta_2 * (lambda_h - exact_solution) * mu_h * ds  # needed if not included as strong BC

# Consistent symmetrization
a += s * delta_0('+') * jump(v, n) * (p('+') - lambda_h("+")) * dS
a += s * delta_0 * dot(v, n) * (p - exact_solution) * ds

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

problem = NonlinearVariationalProblem(F, solution)
# problem = NonlinearVariationalProblem(F, solution, bcs=bc_multiplier)
solver = NonlinearVariationalSolver(problem, solver_parameters=params)
solver.snes.ksp.setConvergenceHistory()
solver.solve()
PETSc.Sys.Print("Solver finished.\n")

u_h, lambda_h = solution.split()
sigma_h = Function(U, name='Velocity')
sigma_h.interpolate(-grad(u_h))
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

# Plotting pressure field exact solution
fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
collection = trisurf(exact_solution, axes=axes, cmap='coolwarm')
fig.colorbar(
    collection,
    orientation="horizontal",
    shrink=0.6, 
    pad=0.1, 
    label="exact solution"
)
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exact solution for pressure")
plt.tight_layout()
plt.savefig("exact_pressure.png")
# plt.show()

fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
collection = trisurf(u_h, axes=axes, cmap='coolwarm')
cbar = fig.colorbar(
    collection,
    orientation="horizontal",
    shrink=0.6, 
    pad=0.1, 
    label="primal variable",
)
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
# plt.clim(0, 1.05)
plt.xlabel("x")
plt.ylabel("y")
plt.tight_layout()
plt.savefig("solution_pressure.png")

# Plotting the mesh
fig, axes = plt.subplots()
collection = triplot(mesh, axes=axes)
# fig.colorbar(collection)
# axes.set_xlim([0, 1])
# axes.set_ylim([0, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("solution_mesh.png")
