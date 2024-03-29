from firedrake import *
import matplotlib.pyplot as plt

parameters["pyop2_options"]["lazy_evaluation"] = False

# Defining the mesh
N = 20
use_quads = True
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads)

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
    # Solver parameters are configured to perform a SVD calculation.
    # Those values are approximations from GMRES iterations in order to
    # estimate the condition number related to FEM system. It is important
    # to set "ksp_gmres_restart" to keep track of max and min singular
    # values approximation through GMRES' Arnoldi iterations.
    'ksp_type': 'gmres',
    'pc_type': 'none',
    'mat_type': 'aij',
    'ksp_max_it': 2000,
    'ksp_monitor_singular_value': None,
    'ksp_gmres_restart': 1000
}

u_h = Function(V)
problem = LinearVariationalProblem(a, L, u_h, bcs=bcs)
solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.snes.ksp.setConvergenceHistory()
solver.solve()

# Retrieving solver information
max_singular_value, min_singular_value = solver.snes.ksp.computeExtremeSingularValues()
condition_number = max_singular_value / min_singular_value
print(f"\n*** Condition number estimate = {condition_number}")

# Plotting solution field
tripcolor(u_h)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
