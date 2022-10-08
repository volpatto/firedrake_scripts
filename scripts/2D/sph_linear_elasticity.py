from firedrake import *
from firedrake.petsc import PETSc
import matplotlib.pyplot as plt

plt.rc('text')
plt.rc('font', size=14)

numel_x, numel_y = 15, 15
use_quads = True
mesh = UnitSquareMesh(numel_x, numel_y, quadrilateral=use_quads)

# Defining the vector function space to primal solution of linear elasticity problem
is_multiplier_continuous = False
displacement_family = 'DQ' if use_quads else 'DG'
degree = 1
degree_multiplier = degree
V = VectorFunctionSpace(mesh, displacement_family, degree)
V_exact = VectorFunctionSpace(mesh, displacement_family, degree + 3)
if is_multiplier_continuous:
    LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
    C0TraceElement = LagrangeElement["facet"]
    T = VectorFunctionSpace(mesh, C0TraceElement)
else:
    trace_family = "HDiv Trace"
    T = VectorFunctionSpace(mesh, trace_family, degree_multiplier)

W = V * T

# Trial and test functions
solution = Function(W)
u, u_hat = split(solution)
v, v_hat  = TestFunctions(W)

# Linear elasticity parameters
E = Constant(1)
nu = Constant(0.499999)
mu = 0.5 * E / (1 + nu)
lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
Id = Identity(mesh.geometric_dimension())

# Mesh entities
n = FacetNormal(mesh)
h = CellDiameter(mesh)
x, y = SpatialCoordinate(mesh)

# Strain and Stress definitions
def epsilon(u):
    return (1. / 2.) * (grad(u) + grad(u).T)


def sigma(u):
    return lambda_ * div(u) * Id + 2 * mu * epsilon(u)


# Exact solution (manufactured)
u_x_exact = 2 * nu * sin(pi * x) * cos(pi * y)
u_y_exact = 2 * (nu - 1) * cos(pi * x) * sin(pi * y)
u_exact = as_vector([u_x_exact, u_y_exact])
u_exact_interpolated = interpolate(u_exact, V_exact)
f = interpolate(-div(sigma(u_exact)), V_exact)

# Plotting the exact solution
fig, axes = plt.subplots(figsize=(8, 8))

collection = quiver(u_exact_interpolated, axes=axes)
fig.colorbar(collection)
axes.set_aspect("equal")

plt.xlabel('x')
plt.ylabel('y')

plt.savefig("solution_displacement_quiver_sph_exact.png")

# Stabilizing parameters
theta_value = -1  # symmetrization parameter (-1 is symmetric, 1 is non-symmetric, 0 is incomplete)
theta = Constant(theta_value)
d = Constant(mesh.geometric_dimension())
k = Constant(degree)
beta_base = lambda beta : beta * (k - 1 + d) * k
beta_0_tilde = Constant(0.5) if use_quads else Constant(2)
beta_0_tilde = beta_0_tilde if theta_value == -1 else Constant(1.5)
beta_0 = beta_base(beta_0_tilde) / h
beta_1_tilde = Constant(1.5) if use_quads else Constant(3)
beta_1_tilde = beta_1_tilde if theta_value == -1 else Constant(0)
beta_1 = beta_base(beta_1_tilde) / h

# *** Variational formulation ***
aP = inner(sigma(u), epsilon(v)) * dx
aP += -inner(dot(sigma(u('+')), n('+')), v('+') - v_hat('+')) * dS
aP += theta * inner(dot(sigma(v('+')), n('+')), u('+') - u_hat('+')) * dS

sP = 2 * mu * beta_0('+') * dot(u('+') - u_hat('+'), v('+') - v_hat('+')) * dS
sP += lambda_ * beta_1('+') * dot(u('+') - u_hat('+'), n('+')) * dot(v('+') - v_hat('+'), n('+')) * dS

# Boundary terms
aP += -inner(dot(sigma(u), n), v - v_hat) * ds
aP += theta * inner(dot(sigma(v), n), u - u_hat) * ds
sP += 2 * mu * beta_0 * dot(u_hat - u_exact, v_hat) * ds
sP += 2 * mu * beta_0 * dot(u - u_exact, v) * ds
sP += lambda_ * beta_1 * dot(u - u_exact, n) * dot(v, n) * ds
sP += lambda_ * beta_1 * dot(u_hat - u_exact, n) * dot(v_hat, n) * ds

# Rearranging as classical a = L form
a = aP + sP
L = dot(f, v) * dx
F = a - L

# Boundary conditions
bcs = DirichletBC(W.sub(1), u_exact, "on_boundary")  # Dirichlet conditions on Lagrange multipliers

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

# problem = NonlinearVariationalProblem(F, solution)
problem = NonlinearVariationalProblem(F, solution, bcs=bcs)
solver = NonlinearVariationalSolver(problem, solver_parameters=params)
solver.solve()
PETSc.Sys.Print("Solver finished.\n")

# Retrieving the solution
u_h, lambda_h = solution.split()
u_h.rename('Displacement', 'label')

# *** Plotting the displacement ***

# Creating the figure object
fig, axes = plt.subplots(figsize=(8, 8))

# Plotting the solution
collection = quiver(u_h, axes=axes)
fig.colorbar(collection)
axes.set_aspect("equal")

# Setting the xy-labels
plt.xlabel('x [L]')
plt.ylabel('y [L]')

# Saving the figure
plt.savefig("solution_displacement_quiver_sph.png")

# Displaying in the screen
plt.show()

# *** Plotting x-axis displacements ***

fig, axes = plt.subplots(figsize=(8, 8))

# Componente x
collection = tripcolor(u_h.sub(0), axes=axes)
fig.colorbar(collection)
axes.set_aspect("equal")

plt.xlabel('x [L]')
plt.ylabel('y [L]')

axes.set_xlim([0, 1])

# plt.show()
plt.savefig("solution_displacement_x_sph.png")

# *** Plotting y-axis displacements ***

fig, axes = plt.subplots(figsize=(8, 8))

# Componente x
collection = tripcolor(u_h.sub(1), axes=axes)
fig.colorbar(collection)
axes.set_aspect("equal")

plt.xlabel('x [L]')
plt.ylabel('y [L]')

axes.set_xlim([0, 1])

# plt.show()
plt.savefig("solution_displacement_y_sph.png")
