from firedrake import *
import matplotlib.pyplot as plt

parameters["pyop2_options"]["lazy_evaluation"] = False

# Defining the mesh
N = 15
use_quads = True
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads)

# Function space declaration
degree = 1
pressure_family = 'CG'
velocity_family = 'CG'
U = VectorFunctionSpace(mesh, velocity_family, degree)
V = FunctionSpace(mesh, pressure_family, degree)
W = U * V

# Trial and test functions
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
solution = Function(W)

# Mesh entities
x, y = SpatialCoordinate(mesh)

# Exact solution
p_exact = sin(2 * pi * x) * sin(2 * pi * y)
exact_solution = Function(V).interpolate(p_exact)
exact_velocity = -grad(p_exact)
exact_solution.rename("Exact pressure", "label")

# Forcing function
f_expression = div(exact_velocity)
f = Function(V).interpolate(f_expression)

# Boundary Conditions
bcs = DirichletBC(W[0], exact_velocity, "on_boundary")

# Least-squares terms
a = inner(u + grad(p), v + grad(q)) * dx
a += div(u) * div(v) * dx
a += inner(curl(u), curl(v)) * dx
L = f * div(v) * dx

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
problem = LinearVariationalProblem(a, L, solution, bcs=bcs)
solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.snes.ksp.setConvergenceHistory()
solver.solve()

sigma_h, u_h = solution.split()
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

# Retrieving solver information
max_singular_value, min_singular_value = solver.snes.ksp.computeExtremeSingularValues()
condition_number = max_singular_value / min_singular_value
print(f"\n*** Condition number estimate = {condition_number}")

# Plotting solution field
tripcolor(u_h)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
