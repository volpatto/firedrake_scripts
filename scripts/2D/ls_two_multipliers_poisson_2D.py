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
is_multiplier_continuous = False
pressure_family = 'DQ' if use_quads else 'DG'
velocity_family = 'DQ' if use_quads else 'DG'
degree = 1
U = VectorFunctionSpace(mesh, velocity_family, degree)
U_exact = VectorFunctionSpace(mesh, velocity_family, degree + 5)
V = FunctionSpace(mesh, pressure_family, degree)
V_exact = FunctionSpace(mesh, pressure_family, degree + 5)
if is_multiplier_continuous:
    LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
    C0TraceElement = LagrangeElement["facet"]
    Tp = FunctionSpace(mesh, C0TraceElement)
    Tu = FunctionSpace(mesh, C0TraceElement)
    T = VectorFunctionSpace(mesh, C0TraceElement, dim=2)
else:
    trace_family = "HDiv Trace"
    Tp = FunctionSpace(mesh, trace_family, degree)
    Tu = FunctionSpace(mesh, trace_family, degree)
    T = VectorFunctionSpace(mesh, trace_family, degree, dim=2)
# T = Tu * Tp
W = U * V * T
# W = U * T * V

# Trial and test functions
solution = Function(W)
u, p, U_b = split(solution)
# u, p, lambda_h = TrialFunctions(W)
v, q, V_b  = TestFunctions(W)

# Trial and test functions
# solution = Function(W)
# u, U_b, p = split(solution)
# # u, p, lambda_h = TrialFunctions(W)
# v, V_b, q  = TestFunctions(W)

# Mesh entities
n = FacetNormal(mesh)
h = CellDiameter(mesh)
x, y = SpatialCoordinate(mesh)

# Exact solution
p_exact = sin(0.5 * pi * x) * sin(0.5 * pi * y)
# p_exact = - (x * x / 2 - x * x * x / 3) * (y * y / 2 - y * y * y / 3)
# p_exact = x * x * x - 3 * x * y * y
exact_solution = Function(V_exact).interpolate(p_exact)
exact_solution.rename("Exact pressure", "label")
u_e = Function(U_exact, name='Exact velocity')
u_e.interpolate(-grad(p_exact))

# Forcing function
f_expression = div(-grad(p_exact))
f = Function(V_exact).interpolate(f_expression)

# Dirichlet BCs
# bc_multiplier = DirichletBC(W.sub(2), p_exact, "on_boundary")

# BCs
p_boundaries = Constant(0.0)
u_projected = u_e

# Stabilizing parameter
delta = Constant(1)
delta_1 = delta  #* h * h
delta_2 = delta  #* h * h
delta_3 = delta * Constant(1)  #* h * h
beta = Constant(1e1 * degree * degree)
alpha = 1 / beta
# delta_4 = beta / h
# # delta_5 = alpha * h * Constant(1e-3)
# delta_5 = alpha * h * h

# Test following prof. Abimael comments
element_size_factor = h * h
delta_4 = beta / element_size_factor * Constant(1)
# delta_3 = 1 / delta_base * h
delta_5 = 1 / beta * element_size_factor * Constant(1)

# Trace unknowns
trace_flux_idx = 0
trace_scalar_idx = 1
lambda_h, sigma_h = U_b[trace_scalar_idx], U_b[trace_flux_idx]
mu_h, tau_h = V_b[trace_scalar_idx], V_b[trace_flux_idx]
sigma_b = sigma_h * n
tau_b = tau_h * n

# Flux Least-squares as in DG
a = delta_1 * inner(u + grad(p), v + grad(q)) * dx

# # Flux least-squares
# Comments: For this expanded form, when using every parameter as one (note that edge terms in flux contribution should match
# with transmission condition weighting), super convergence was achieved for triangles for degree = 1.
# It is important to note that, in such a case, beta is not zero, so u_hat edge stabilization should be included.
# a = (
#     (inner(u, v) - q * div(u) - p * div(v) + inner(grad(p), grad(q)))
#     * delta_1
#     * dx
# )
# # These terms below are unsymmetric
# a += delta_1("+") * jump(sigma_b, n=n) * mu_h("+") * dS
# a += delta_1 * dot(sigma_b, n) * mu_h * ds
# a += delta_1("+") * lambda_h("+") * jump(tau_b, n=n) * dS
# a += delta_1 * lambda_h * dot(tau_b, n) * ds

# Mass balance least-square
a += delta_2 * div(u) * div(v) * dx
L = delta_2 * f * div(v) * dx

# Irrotational least-squares
a += delta_3 * inner(curl(u), curl(v)) * dx

# Transmission conditions
enable_transmission_conditions = 0
# a += Constant(1) * sigma_h("+") * tau_h("+") * dS
a += Constant(enable_transmission_conditions) * delta_4("+") * dot(jump(lambda_h, n=n), jump(mu_h, n=n)) * dS
# a += Constant(enable_transmission_conditions) * delta_5("+") * jump(sigma_b, n=n) * jump(tau_b, n=n) * dS
a += Constant(enable_transmission_conditions) * delta_5("+") * dot(jump(sigma_h, n=n), jump(tau_h, n=n)) * dS
a += Constant(enable_transmission_conditions) * delta_5 * sigma_h * tau_h * ds
a += Constant(enable_transmission_conditions) * delta_5 * dot(u_e, n) * tau_h * ds

# Boundary identifications
enable_boundary_residuals = 1
a += Constant(enable_boundary_residuals) * delta_4("+") * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
a += Constant(enable_boundary_residuals) * delta_4 * (p - lambda_h) * q * ds
a += Constant(enable_boundary_residuals) * delta_4 * (lambda_h - p_exact) * mu_h * ds
a += Constant(enable_boundary_residuals) * delta_4("+") * dot((u("+") - sigma_b("+")), (v("+") - tau_b("+"))) * dS
a += Constant(enable_boundary_residuals) * delta_4 * dot((u - u_e), v) * ds
a += Constant(enable_boundary_residuals) * delta_4 * dot((sigma_b - u_e), tau_b) * ds

a += Constant(enable_boundary_residuals) * delta_5("+") * dot(u("+") - sigma_b("+"), n("+")) * dot(v("+") - tau_b("+"), n("+")) * dS
# a += delta_5("+") * (dot(u("+"), n("+")) - sigma_h("+")) * (dot(v("+"), n("+")) - tau_h("+")) * dS
# a += delta_5("+") * jump(u - sigma_b, n=n) * jump(v - tau_b, n=n) * dS
a += Constant(enable_boundary_residuals) * delta_5 * dot((u - sigma_b), n) * dot(v, n) * ds
a += Constant(enable_boundary_residuals) * delta_5 * dot((sigma_b - u_e), n) * tau_h * ds
# a += delta_5 * (dot(u, n) - dot(u_e, n)) * dot(v, n) * ds
# a += delta_5 * (sigma_h - dot(u_e, n)) * tau_h * ds

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
        'ksp_view': None,
        "pc_type": "lu",
        "ksp_monitor": None,
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_4": "2",
        "ksp_monitor_true_residual": None
    },
}

problem = NonlinearVariationalProblem(F, solution)
# problem = NonlinearVariationalProblem(F, solution, bcs=bc_multiplier)
solver = NonlinearVariationalSolver(problem, solver_parameters=params)
solver.snes.ksp.setConvergenceHistory()
solver.solve()
PETSc.Sys.Print("Solver finished.\n")

u_h, p_h, U_b = solution.split()
# u_h, U_b, p_h = solution.split()
sigma_h, lambda_h = U_b[trace_flux_idx], U_b[trace_scalar_idx]
# u_h, p_h, sigma_h, lambda_h = solution.split()
u_h.rename('Velocity', 'label')
p_h.rename('Pressure', 'label')

# Plotting velocity field exact solution
fig, axes = plt.subplots()
collection = quiver(u_e, axes=axes, cmap='coolwarm')
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
collection = quiver(u_h, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("solution_velocity.png")
# plt.show()

# Plotting pressure field numerical solution
fig, axes = plt.subplots()
collection = tripcolor(p_h, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("solution_pressure.png")
# plt.show()
