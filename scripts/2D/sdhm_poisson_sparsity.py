import copy
from firedrake import *
import matplotlib
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc

matplotlib.use('Agg')
my_cmap = copy.copy(plt.cm.get_cmap("winter"))
my_cmap.set_bad(color="lightgray")


def print(content_to_print):
    return PETSc.Sys.Print(content_to_print)


def plot_matrix_hybrid_full(a_form, bcs=[], **kwargs):
    """Provides a plot of a full hybrid-mixed matrix."""
    fig, ax = plt.subplots(1, 1)

    A = assemble(a_form, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle

    total_size = petsc_mat.getSize()
    f0_size = A.M[0, 0].handle.getSize()
    f1_size = A.M[1, 1].handle.getSize()

    Mij = PETSc.Mat()
    petsc_mat.convert("aij", Mij)

    n, m = total_size
    Mnp = np.array(Mij.getValues(range(n), range(m)))
    Am = np.ma.masked_values(Mnp, 0, rtol=1e-13)

    # Plot the matrix
    plot = ax.matshow(Am, **kwargs)

    # Remove axis ticks and values
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.axhline(y=f0_size[0] - 0.5, color="k")
    ax.axvline(x=f0_size[0] - 0.5, color="k")
    ax.axhline(y=f0_size[0] + f1_size[0] - 0.5, color="k")
    ax.axvline(x=f0_size[0] + f1_size[0] - 0.5, color="k")

    return plot


def plot_matrix_hybrid_multiplier_spp(a_form, bcs=[], **kwargs):
    """Provides a plot of a condensed hybrid-mixed matrix for single scale problems."""
    fig, ax = plt.subplots(1, 1)

    _A = Tensor(a_form)
    A = _A.blocks
    S = A[2, 2] - A[2, :2] * A[:2, :2].inv * A[:2, 2]
    Smat = assemble(S, bcs=bcs)

    petsc_mat = Smat.M.handle
    total_size = petsc_mat.getSize()
    Mij = PETSc.Mat()
    petsc_mat.convert("aij", Mij)

    n, m = total_size
    Mnp = np.array(Mij.getValues(range(n), range(m)))
    Am = np.ma.masked_values(Mnp, 0, rtol=1e-13)

    # Plot the matrix
    plot = ax.matshow(Am, **kwargs)
    # Below there is the spy alternative
    # plot = plt.spy(Am, **kwargs)

    # Remove axis ticks and values
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return plot


parameters["pyop2_options"]["lazy_evaluation"] = False
PETSc.Log.begin()

# Defining the mesh
N = 5
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
u, p, lambda_h = TrialFunctions(W)
v, q, mu_h = TestFunctions(W)

# Mesh entities
n = FacetNormal(mesh)
x, y = SpatialCoordinate(mesh)
h = CellDiameter(mesh)

# Model parameters
k = Constant(1.0)
mu = Constant(1.0)
rho = Constant(0.0)
g = Constant((0.0, 0.0))

# Exact solution and source term projection
p_exact = sin(2 * pi * x) * sin(2 * pi * y)
sol_exact = Function(V).interpolate(p_exact)
sol_exact.rename("Exact pressure", "label")
sigma_e = Function(U, name="Exact velocity")
sigma_e.project(-(k / mu) * grad(p_exact))
# plot(sigma_e)
quiver(sigma_e)
source_expr = div(-(k / mu) * grad(p_exact))
f = Function(V).interpolate(source_expr)
# plot(sol_exact)
tripcolor(sol_exact)
plt.axis("off")

# BCs
p_boundaries = Constant(0.0)
v_projected = sigma_e
bc_multiplier = DirichletBC(W.sub(2), Constant(0.0), "on_boundary")

# Hybridization parameter
beta_0 = Constant(1.0e-15)
beta = beta_0 / h
beta_avg = beta_0 / h("+")

# Mixed classical terms
a = (dot((mu / k) * u, v) - div(v) * p - q * div(u)) * dx
L = -f * q * dx - dot(rho * g, v) * dx
# Stabilizing terms
a += -0.5 * inner((k / mu) * ((mu / k) * u + grad(p)), (mu / k) * v + grad(q)) * dx
a += 0.5 * (mu / k) * div(u) * div(v) * dx
a += 0.5 * inner((k / mu) * curl((mu / k) * u), curl((mu / k) * v)) * dx
L += 0.5 * (mu / k) * f * div(v) * dx
# Hybridization terms
a += lambda_h("+") * dot(v, n)("+") * dS + mu_h("+") * dot(u, n)("+") * dS
a += beta_avg * (lambda_h("+") - p("+")) * (mu_h("+") - q("+")) * dS

F = a - L

plot_matrix_hybrid_multiplier_spp(a, bcs=bc_multiplier, cmap=my_cmap)
plt.savefig("condensed_matrix.png")
# plt.show()

plot_matrix_hybrid_full(a, bcs=bc_multiplier, cmap=my_cmap)
plt.savefig("full_matrix.png")
# plt.show()

print("\n*** DoF = %i" % W.sub(2).dim())
