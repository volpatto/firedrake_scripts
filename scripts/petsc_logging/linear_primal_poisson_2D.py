from firedrake import *
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc


def print(content_to_print):
    return PETSc.Sys.Print(content_to_print)


parameters["pyop2_options"]["lazy_evaluation"] = False
PETSc.Log.begin()

# Defining the mesh
N = 20
use_quads = False
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads)
comm = mesh.comm

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Mesh coordinates
x, y = SpatialCoordinate(mesh)

# Exact solution
p_exact = sin(2 * pi * x) * sin(2 * pi * y)
exact_solution = Function(V).interpolate(p_exact)
exact_solution.rename("Exact pressure", "label")

# Forcing function
f_expression = div(-grad(p_exact))
f = Function(V).interpolate(f_expression)

# Dirichlet BCs
bcs = DirichletBC(V, 0.0, "on_boundary")

# Variational form
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# Solving the system
solver_parameters = {
    'ksp_monitor': None,
    'ksp_view': None,
    'ksp_type': 'gmres',
    'pc_type': 'none',
    'mat_type': 'aij',
    'ksp_max_it': 2000
}
u_h = Function(V)
problem = LinearVariationalProblem(a, L, u_h, bcs=bcs)
solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.snes.ksp.setConvergenceHistory()
solver.solve()

# Plotting solution field
tripcolor(u_h)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
