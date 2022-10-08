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
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads)
comm = mesh.comm

# Function space declaration
is_multiplier_continuous = False
pressure_family = 'DQ' if use_quads else 'DG'
velocity_family = 'DQ' if use_quads else 'DG'
degree = 2
U = VectorFunctionSpace(mesh, velocity_family, degree)
V = FunctionSpace(mesh, pressure_family, degree)
V_exact = FunctionSpace(mesh, pressure_family, degree + 3)
if is_multiplier_continuous:
    LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
    C0TraceElement = LagrangeElement["facet"]
    T = FunctionSpace(mesh, C0TraceElement)
else:
    trace_family = "HDiv Trace"
    T = FunctionSpace(mesh, trace_family, degree)

    LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree + 1)
    C0TraceElement = LagrangeElement["facet"]
    Q = FunctionSpace(mesh, C0TraceElement)
W = V * T * Q

# Trial and test functions
solution = Function(W)
p, lambda_h, phi_h = split(solution)
# p, lambda_h, phi_h = TrialFunctions(W)
q, mu_h, psi_h  = TestFunctions(W)

# Mesh entities
n = FacetNormal(mesh)
h = CellDiameter(mesh)
x, y = SpatialCoordinate(mesh)

# Exact solution
# p_exact = sin(2 * pi * x) * sin(2 * pi * y)
# p_exact = sin(0.5 * pi * x) * sin(0.5 * pi * y)
p_exact = x * x * x - 3 * x * y * y
# p_exact = 1 + x + y
# p_exact = 1 + x + y + x * y + x * x - y * y + x * x * x - 3 * x * y * y
exact_solution = Function(V_exact).interpolate(p_exact)
exact_solution.rename("Exact pressure", "label")
sigma_e = Function(U, name='Exact velocity')
sigma_e.interpolate(-grad(p_exact))

# Forcing function
f_expression = div(-grad(p_exact))
f = Function(V_exact).interpolate(f_expression)

# Dirichlet BCs
bc_multiplier = DirichletBC(W.sub(1), p_exact, "on_boundary")

# Hybridization parameter
beta_0 = Constant(0 * degree * degree)  # should not be zero when used with LS terms
beta = beta_0 / h
# beta = beta_0

# Stabilizing parameter
# delta_base = h * h
delta_base = Constant(1)
delta_0 = Constant(1)
delta_1 = delta_base * Constant(1)
# delta_2 = delta_base * Constant(1) / h
edge_base = Constant(1e1 * degree * degree)
delta_2 = edge_base / h * Constant(1e0)
# delta_3 = delta_2
delta_3 = 1 / edge_base * h * Constant(1e0)
# delta_2 = beta
# delta_2 = delta_1 * Constant(1)  # so far this is the best combination

# Flux variables
u = -grad(p)
v = -grad(q)

# Symmetry parameter: s = 1 (symmetric) or s = -1 (unsymmetric). Disable with 0.
s = Constant(0)

# Numerical flux trace
# u_hat = u + beta * (p - lambda_h) * n
# u_hat = u

# Classical term
a = delta_0 * dot(grad(p), grad(q)) * dx + delta_0('+') * phi_h("+") * q("+") * dS
# a += delta_0 * dot(u_hat, n) * q * ds
# a += delta_0 * (phi_h - dot(sigma_e, n)) * q * ds
L = delta_0 * f * q * dx

# Mass balance least-square
a += delta_1 * div(u) * div(v) * dx
# a += delta_1 * inner(curl(u), curl(v)) * dx
L += delta_1 * f * div(v) * dx

# Hybridization terms
# a += psi_h("+") * phi_h("+") * dS
# a += mu_h * (lambda_h - exact_solution) * ds
# a += mu_h * dot(u_hat - grad(exact_solution), n) * ds

# Least-Squares on constrains
a += delta_2("+") * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
a += delta_3("+") * (jump(u, n=n) - phi_h("+")) * (jump(v, n=n) - psi_h("+")) * dS
# a += delta_2 * (p - exact_solution) * (q - mu_h) * ds  # needed if not included as strong BC
a += delta_2 * (p - exact_solution) * q * ds  # needed if not included as strong BC
a += delta_2 * (lambda_h - exact_solution) * mu_h * ds  # needed if not included as strong BC
# a += delta_3 * (dot(u, n) - dot(sigma_e, n)) * dot(v, n) * ds
a += delta_3 * (phi_h - dot(sigma_e, n)) * psi_h * ds

# Consistent symmetrization
a += s * delta_0 * psi_h("+") * (p('+') - lambda_h("+")) * dS
a += s * delta_0 * psi_h * (p - exact_solution) * ds

# Symmetrization based on flux continuity (alternative) -- doesn't work
# a += -lambda_h('+') * jump(v, n) * dS
# L += -exact_solution * dot(v, n) * ds

F = a - L
# A = lhs(F)
# b = rhs(F)

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
    "pc_sc_eliminate_fields": "0, 2",
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
# # problem = LinearVariationalProblem(a, L, solution, bcs=[bc_multiplier])
# problem = LinearVariationalProblem(A, b, solution, bcs=[])
# solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
# solver.snes.ksp.setConvergenceHistory()
# solver.solve()
# print("Solver finished.\n")

u_h, lambda_h, phi_h = solution.split()
sigma_h = Function(U, name='Velocity')
sigma_h.interpolate(-grad(u_h))
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
# triplot(mesh, axes=axes)
fig.colorbar(collection)
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("solution_pressure.png")
# plt.show()

# Plotting the mesh
fig, axes = plt.subplots()
collection = triplot(mesh, axes=axes)
# fig.colorbar(collection)
# axes.set_xlim([0, 1])
# axes.set_ylim([0, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("solution_mesh.png")
