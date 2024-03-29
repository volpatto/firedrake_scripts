from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD

try:
    import matplotlib.pyplot as plt

    plt.rcParams["contour.corner_mask"] = False
    plt.close("all")
except:
    warning("Matplotlib not imported")

nx, ny = 20, 20
quads = True
mesh = UnitSquareMesh(nx, ny, quadrilateral=quads)

degree = 1
k_plus = 0
primal_family = "DQ" if quads else "DG"
U = FunctionSpace(mesh, primal_family, degree + k_plus)
V = VectorFunctionSpace(mesh, "CG", degree + k_plus)
LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
C0TraceElement = LagrangeElement["facet"]
T = FunctionSpace(mesh, C0TraceElement)
W = U * T

# Trial and test functions
solution = Function(W)
u, lambda_h = split(solution)
v, mu_h = TestFunction(W)

# Mesh entities
n = FacetNormal(mesh)
x, y = SpatialCoordinate(mesh)

# Exact solution and source term projection
p_exact = sin(2 * pi * x) * sin(2 * pi * y)
sol_exact = Function(U).interpolate(p_exact)
sol_exact.rename("Exact pressure", "label")
sigma_e = Function(V, name="Exact velocity")
sigma_e.project(-grad(p_exact))
source_expr = div(-grad(p_exact))
f = Function(U).interpolate(source_expr)

# BCs
p_boundaries = Constant(0.0)
bc_multiplier = DirichletBC(W.sub(1), p_boundaries, "on_boundary")

# DG parameter
s = Constant(1.0)
beta = Constant(32.0)
h = CellDiameter(mesh)
h_avg = avg(h)

# Classical term
a = dot(grad(u), grad(v)) * dx
L = f * v * dx
# Hybridization terms
a += s * dot(grad(v), n)("+") * (u("+") - lambda_h("+")) * dS
a += -dot(grad(u), n)("+") * (v("+") - mu_h("+")) * dS
a += (beta / h_avg) * (u("+") - lambda_h("+")) * (v("+") - mu_h("+")) * dS
# Boundary terms
# a += -dot(vel_projected, n) * v * ds  # How to set this bc??
a += (beta / h) * (u - p_boundaries) * v * ds  # is this necessary?
L += s * dot(grad(v), n) * p_boundaries * ds

F = a - L

#  Solving SC below
PETSc.Sys.Print("*******************************************\nSolving...\n")
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
    },
}

problem = NonlinearVariationalProblem(F, solution, bcs=bc_multiplier)
solver = NonlinearVariationalSolver(problem, solver_parameters=params)
solver.solve()

PETSc.Sys.Print("Solver finished.\n")

# Gathering solution
u_h, lambda_h = solution.split()
u_h.rename("Solution", "label")

# Post-processing solution
sigma_h = Function(V, name="Projected velocity")
sigma_h.project(-grad(u_h))

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
collection = tripcolor(sol_exact, axes=axes, cmap='coolwarm')
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


print("\n*** DoF = %i" % W.dim())
