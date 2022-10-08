from json.encoder import py_encode_basestring_ascii
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
use_quads = True
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads)

# Function space declaration
pressure_family = 'DQ' if use_quads else 'DG'
velocity_family = 'DQ' if use_quads else 'DG'
degree = 1
U = VectorFunctionSpace(mesh, velocity_family, degree)  # just for post-processing
V = FunctionSpace(mesh, pressure_family, degree)
V_exact = FunctionSpace(mesh, pressure_family, degree + 3)

# Trial and test functions
p = TrialFunction(V)
q = TestFunction(V)

# Mesh entities
n = FacetNormal(mesh)
h = CellDiameter(mesh)
x, y = SpatialCoordinate(mesh)

# Reaction and Diffusion coefficients
K = Constant(1e-8)
D = Constant(1)

# Exact solution
p_exact = conditional(
    And(gt(x, 0.0), lt(x, 0.5)),
    conditional(And(gt(y, 0.0), lt(y, 1.0)), Constant(1), Constant(0)),
    Constant(0)
)
exact_solution = Function(V_exact).interpolate(p_exact)
exact_solution.rename("Exact pressure", "label")
sigma_e = Function(U, name='Exact velocity')
sigma_e.project(-K * grad(p_exact))

# Forcing function
f_expression = div(-K * grad(p_exact)) + D * p_exact
# f_expression = p_exact
f = Function(V).project(f_expression)

# GGLS stabilization parameters
alpha = (D * h * h) / (6.0 * K)
eps = conditional(ge(alpha, 8), 1, conditional(ge(alpha, 1), 0.064 * alpha + 0.49, 0))
tau = (eps * (h * h)) / (6.0 * D)

# Stabilizing parameter
# Combinations that work:
# 1) LS only: delta_1, delta_2, and delta_3 activated (delta_4 not required! both tri and quads)
# 2) DG-IP + GLS: Only DG-IP terms and delta_4 activated. (multiply delta_4 by 100 for triangles)
# 3) All terms activated. (multiply delta_4 by 100 for triangles)
# 4) DG-IP + LS: delta_0, delta_1, delta_2, and delta_3 activated. (both triangles and quads)
# 5) DG-IP: delta_0 only. (both triangles and quads)
element_size_factor = h * h
penalty_constant = K
# delta_base = h * h
delta_base = Constant(penalty_constant * degree * degree)
# delta_base = Constant(penalty_constant)
enable_dg_ip = Constant(0)  # enable (1) or disable (0)
delta_0 = delta_base / delta_base * enable_dg_ip
delta_1 = h * h * Constant(1)
# delta_1 = Constant(1)
delta_2 = delta_base / element_size_factor * Constant(1)
delta_3 = 1 / delta_base * element_size_factor * Constant(1)
enable_gls = Constant(0)  # enable (1) or disable (0)
delta_4 = enable_gls * tau / K * Constant(1e2)  # this 100 is empirical

# Flux variables
u = -K * grad(p)
v = -K * grad(q)

# Classical DG-IP term
a = delta_0 * dot(K * grad(p), grad(q)) * dx + delta_0 * D * p * q * dx
L = delta_0 * f * q * dx

# Symmetry term. Choose if the method is SIPG (-1) or NIPG (1)
s = Constant(-1)

# DG edge terms
a += s * delta_0 * dot(jump(p, n), avg(v)) * dS
a += -delta_0 * dot(avg(u), jump(q, n)) * dS

# Mass balance least-square
a += delta_1 * (div(u) + D * p) * (div(v) + D * q) * dx
L += delta_1 * f * (div(v) + D * q) * dx

# Mass balance gradient least-square
a += delta_4 * inner(grad(div(u) + D * p), grad(div(v) + D * q)) * dx
L += delta_4 * inner(grad(f), grad(div(v) + D * q)) * dx

# Hybridization terms
a += avg(delta_2) * dot(jump(p, n=n), jump(q, n=n)) * dS
a += delta_2 * (p - p_exact) * q * ds
a += avg(delta_3) * jump(u, n=n) * jump(v, n=n) * dS
# Flux should not be imposed at Dirichlet condition boundaries
# a += delta_3 * dot(u, n) * dot(v, n) * ds
# L += delta_3 * dot(sigma_e, n) * dot(v, n) * ds

# DG-IP Weak boundary conditions (not required, already imposed by LS terms)
beta0 = Constant(enable_dg_ip * penalty_constant * degree * degree * 1)
beta = beta0 / h  # Nitsche term
a += s * delta_0 * dot(p * n, v) * ds - delta_0 * dot(u, q * n) * ds
a += delta_0 * beta * p * q * ds
L += delta_0 * beta * p_exact * q * ds

# Ensuring that the formulation is properly decomposed in LHS and RHS
F = a - L
a_form = lhs(F)
L_form = rhs(F)

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
problem = LinearVariationalProblem(a_form, L_form, solution, bcs=[])
solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.solve()

u_h = solution
sigma_h = Function(U, name='Velocity')
sigma_h.project(-K * grad(u_h))
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

# Plotting pressure field exact solution
fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
# fig, axes = plt.subplots()
collection = trisurf(exact_solution, axes=axes, cmap='coolwarm')
# collection = tripcolor(exact_solution, axes=axes, cmap='coolwarm')
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
# fig, axes = plt.subplots()
collection = trisurf(u_h, axes=axes, cmap='coolwarm', vmin=0)
# collection = tripcolor(u_h, axes=axes, cmap='coolwarm', vmin=0)
cbar = fig.colorbar(
    collection,
    orientation="horizontal",
    shrink=0.6, 
    pad=0.1, 
    label="primal variable"
)
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
# plt.clim(0, 1)
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
