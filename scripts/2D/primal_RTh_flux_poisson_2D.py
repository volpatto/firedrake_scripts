from firedrake import *
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc

def print(content_to_print):
    return PETSc.Sys.Print(content_to_print)


PETSc.Sys.popErrorHandler()

# Defining the mesh
N = 10
use_quads = False
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads)
comm = mesh.comm

# Function space declaration
degree = 1
if use_quads:
    hdiv_family = 'RTCF'
    pressure_family = 'DQ'
    # RTk = FiniteElement("Raviart-Thomas", quadrilateral, degree + 1, variant="integral")
else:
    hdiv_family = 'RT'
    pressure_family = 'DG'
    # RTk = FiniteElement("Raviart-Thomas", triangle, degree + 1, variant="integral")

trace_family = "HDiv Trace"
is_multiplier_continuous = True
U = FunctionSpace(mesh, hdiv_family, degree)
# U = VectorFunctionSpace(mesh, pressure_family, degree - 1)
# RTk = FiniteElement("Raviart-Thomas", triangle, degree + 1, variant="integral")
# U = FunctionSpace(mesh, RTk)
V = FunctionSpace(mesh, pressure_family, degree)
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
# p, lambda_h = TrialFunctions(W)
q, mu_h  = TestFunctions(W)

# Mesh entities
n = FacetNormal(mesh)
h = CellDiameter(mesh)
x, y = SpatialCoordinate(mesh)

# Exact solution
# p_exact = sin(2 * pi * x) * sin(2 * pi * y)
p_exact = x * x * x - 3 * x * y * y
exact_solution = Function(V).interpolate(p_exact)
exact_solution.rename("Exact pressure", "label")
sigma_e = Function(U, name='Exact velocity')
sigma_e.project(-grad(p_exact))
# sigma_n = Function(T).interpolate(dot(sigma_e, n))

# Forcing function
f_expression = div(-grad(p_exact))
f = Function(V).interpolate(f_expression)

# Dirichlet BCs (not necessary to impose in this case)
# bc_multiplier = DirichletBC(W[1], project(dot(-sigma_e, n), W[1]), "on_boundary")

# Classical terms
a = inner(grad(p), grad(q)) * dx + lambda_h('+') * q('+') * dS
L = f * q * dx

# Ewing, Wang and Yang stabilization terms
beta_0 = Constant(1e1 * degree * degree)
alpha = 1 / beta_0 * h
# alpha = beta_0
a += -alpha('+') * (jump(grad(p), n) - lambda_h('+')) * jump(grad(q), n) * dS

# Transmission condition
s = Constant(-1)
a += s * p('+') * mu_h('+') * dS + alpha('+') * (lambda_h('+') - jump(grad(p), n)) * mu_h('+') * dS
# a += (lambda_h('+') + lambda_h('-')) * mu_h('+') * dS
# a += avg(lambda_h) * mu_h('+') * dS

# Boundary terms
# a += alpha * (dot(grad(p), n) - lambda_h) * dot(grad(q), n) * ds
a += -alpha * lambda_h * dot(grad(q), n) * ds
L += -alpha * dot(grad(exact_solution), n) * dot(grad(q), n) * ds

# a += alpha * (lambda_h - dot(grad(p), n)) * mu_h * ds
a += alpha * lambda_h * mu_h * ds
L += alpha * dot(grad(exact_solution), n) * mu_h * ds

a += -lambda_h * q * ds
L += -dot(grad(exact_solution), n) * q * ds
a += s * p * mu_h * ds
L += s * exact_solution * mu_h * ds

F = a - L
# a = rhs(F)
# L = lhs(F)

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
    "pc_sc_eliminate_fields": "1",
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

# solver_parameters = {
#     "ksp_monitor": None,
#     "mat_type": "aij",
#     "ksp_type": "preonly",
#     "pc_type": "lu",
#     "pc_factor_mat_solver_type": "mumps",
# }
# solver_parameters = {
#     "ksp_monitor": None,
# }
# # solution = Function(W)
# solve(F == 0, solution, solver_parameters=solver_parameters)
# # nullspace = MixedVectorSpaceBasis(W, [W[0], VectorSpaceBasis(constant=True)])
# problem = LinearVariationalProblem(a, L, solution, bcs=[])
# solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
# # problem = NonlinearVariationalProblem(F, solution, bcs=bc_multiplier)
# # problem = NonlinearVariationalProblem(F, solution)
# # solver = NonlinearVariationalSolver(problem)
# solver.snes.ksp.setConvergenceHistory()
# solver.solve()
# print("Solver finished.\n")

u_h, lambda_h = solution.split()
sigma_h = Function(U, name='Velocity')
sigma_h.project(-grad(u_h))
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

