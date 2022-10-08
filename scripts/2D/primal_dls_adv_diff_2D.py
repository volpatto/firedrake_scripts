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
use_quads = True  # quads perform better
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads)

# Function space declaration
pressure_family = 'DQ' if use_quads else 'DG'
velocity_family = 'DQ' if use_quads else 'DG'
degree = 1  # should be greater/equal to 2 for pure LS form
U = VectorFunctionSpace(mesh, velocity_family, degree)
V = FunctionSpace(mesh, pressure_family, degree)

# Trial and test functions
p = TrialFunction(V)
q = TestFunction(V)

# Mesh entities
n = FacetNormal(mesh)
h = CellDiameter(mesh)
x, y = SpatialCoordinate(mesh)

# Reaction and Diffusion coefficients
K = Constant(0)
D = Constant(0)

# Velocity field
b = Function(U)
b.project(as_vector([1.0, 0.0]))

# Auxiliary upwind parameters
is_inward = conditional(
    dot(b, n) < 0,  # condition
    1,  # true
    0  # false
)
is_outward = conditional(
    dot(b, n) > 0,  # condition
    1,  # true
    0  # false
)

# Exact solution
p_exact = conditional(le(x, 0.9), x, 0)
exact_solution = Function(V).interpolate(p_exact)
exact_solution.rename("Exact pressure", "label")
sigma_e = Function(U, name='Exact velocity')
sigma_e.project(-grad(p_exact))

# Forcing function
f_expression = div(-K * grad(p_exact)) + D * p_exact + dot(b, grad(p_exact))
f = Function(V).interpolate(f_expression)

# *** Adding residual stabilizing terms ***
# Stabilizing parameters (based on Franca et al. 1992)
if degree == 1:
    m_k = 1.0 / 3.0
elif degree == 2 and not use_quads:
    m_k = 1.0 / 12.
elif degree == 2 and use_quads:
    m_k = 1.0 / 24.
else:
    raise ValueError("The employed mesh currently has no stabilization parameters available.")

# Peclet number
h_k = sqrt(2) * CellVolume(mesh) / CellDiameter(mesh)
b_norm = sqrt(dot(b, b))
Pe_k = m_k * b_norm * h_k / (2.0 * K)

# Stabilizing parameter
# delta_base = h * h
element_size_factor = h * h
penalty_constant = 1e0
delta_base = Constant(penalty_constant * degree * degree)
# delta_base = Constant(1e0 * degree * degree)
enable_dg_ip = Constant(0)  # enable (1) or disable (0)
delta_0 = delta_base / delta_base * enable_dg_ip
delta_1 = Constant(1)
# delta_2 = delta_base / h
# Testar esses valores, Abimael acha que é ao cubo. Testar combinações
# delta_2 = delta_base / (h * h * h)
# delta_2 = delta_base / (h * h)
# delta_2 = delta_base / h
delta_2 = delta_base / element_size_factor * Constant(1)
# delta_3 = 1 / delta_base * h
delta_3 = 1 / delta_base * element_size_factor * Constant(1)
# delta_3 = delta_2

# Flux variables
u = -K * grad(p) + b * p
v = -K * grad(q) + b * q

# Residual definition
Lp = div(u) + D * p 
Lq = div(v) + D * q

# Classical DG-IP term
a = delta_0 * dot(K * grad(p), grad(q)) * dx + delta_0 * D * p * q * dx
a += delta_0 * dot(b, grad(p)) * q * dx
L = delta_0 * f * q * dx

# Symmetry term. Choose if the method is SIPG (-1) or NIPG (1)
s = Constant(-1)

# DG edge terms
a += s * delta_0 * dot(jump(p, n), avg(v)) * dS
a += -delta_0 * dot(avg(u), jump(q, n)) * dS

# Mass balance least-square
a += delta_1 * Lp * Lq * dx
L += delta_1 * f * Lq * dx

# Hybridization terms
a += avg(delta_2) * dot(jump(p, n=n), jump(q, n=n)) * dS
# a += delta_2 * is_inward * (p - p_exact) * abs(dot(b, n)) * q * ds
a += delta_2 * (p - p_exact) * abs(dot(b, n)) * q * ds
a += avg(delta_3) * jump(u, n=n) * jump(v, n=n) * dS
# Flux should not be imposed at Dirichlet condition boundaries
# a += delta_3 * dot(u, n) * dot(v, n) * ds
# L += delta_3 * dot(sigma_e, n) * dot(v, n) * ds

# DG-IP Weak boundary conditions (not required, already imposed by LS terms)
beta0 = Constant(enable_dg_ip * penalty_constant * degree * degree * 1)
beta = beta0 / h  # Nitsche term
a += s * delta_0 * dot(p * n, v) * ds - delta_0 * dot(u, q * n) * ds
a += delta_0 * beta * is_outward * p * q * abs(dot(b, n)) * ds
a += delta_0 * beta * is_inward * (p - exact_solution) * q * abs(dot(b, n)) * ds
# a += delta_0 * beta * p * q * ds
# L += delta_0 * beta * exact_solution * q * ds

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
sigma_h.project(-grad(u_h))
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

# Plotting velocity field exact solution
fig, axes = plt.subplots()
triplot(mesh, axes=axes)
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
triplot(mesh, axes=axes)
collection = quiver(sigma_h, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("solution_velocity.png")
# plt.show()

# Plotting pressure field numerical solution
# fig, axes = plt.subplots()
# collection = tripcolor(u_h, axes=axes, cmap='coolwarm')
# fig.colorbar(collection)
# triplot(mesh, axes=axes)
# axes.set_xlim([0, 1])
# axes.set_ylim([0, 1])
# plt.xlabel("x")
# plt.ylabel("y")
# plt.savefig("solution_pressure.png")
# # plt.show()

fig, axes = plt.subplots(subplot_kw={"projection": "3d"})
collection = trisurf(u_h, axes=axes, cmap='coolwarm', vmin=0)
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
