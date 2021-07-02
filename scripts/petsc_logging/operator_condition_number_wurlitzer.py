from io import StringIO

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from wurlitzer import pipes, STDOUT

parameters["pyop2_options"]["lazy_evaluation"] = False


def solve_poisson_cg(num_elements_x, num_elements_y, degree=1, use_quads=False):
    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
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
    # The below code only uses SVD to estimate the condition number.
    # Caution is needed since it can be computationally expensive.
    solver_parameters = {
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'svd',
        'pc_svd_monitor': None,
        'ksp_monitor_singular_value': None,
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij'
    }
    u_h = Function(V)
    problem = LinearVariationalProblem(a, L, u_h, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.snes.ksp.setConvergenceHistory()
    solver.solve()

    return u_h


def solve_poisson_ls(num_elements_x, num_elements_y, degree=1, use_quads=False):
    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
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
    exact_solution.rename("Exact pressure", "label")

    # Forcing function
    f_expression = div(-grad(p_exact))
    f = Function(V).interpolate(f_expression)

    # Boundaries: Left (1), Right (2), Bottom(3), Top (4)
    vx = -2 * pi * cos(2 * pi * x) * sin(2 * pi * y)
    vy = -2 * pi * sin(2 * pi * x) * cos(2 * pi * y)

    bc1 = DirichletBC(W[0], as_vector([vx, 0.0]), 1)
    bc2 = DirichletBC(W[0], as_vector([vx, 0.0]), 2)
    bc3 = DirichletBC(W[0], as_vector([0.0, vy]), 3)
    bc4 = DirichletBC(W[0], as_vector([0.0, vy]), 4)
    bcs = [bc1, bc2, bc3, bc4]

    # Least-squares terms
    a = inner(u + grad(p), v + grad(q)) * dx
    a += div(u) * div(v) * dx
    a += inner(curl(u), curl(v)) * dx
    L = f * div(v) * dx

    # Solving the system
    # The below code only uses SVD to estimate the condition number.
    # Caution is needed since it can be computationally expensive.
    solver_parameters = {
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'svd',
        'pc_svd_monitor': None,
        'ksp_monitor_singular_value': None,
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij'
    }
    problem = LinearVariationalProblem(a, L, solution, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.snes.ksp.setConvergenceHistory()
    solver.solve()

    sigma_h, u_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    u_h.rename('Pressure', 'label')

    return u_h, sigma_h


def solve_poisson_cgls(num_elements_x, num_elements_y, degree=1, use_quads=False):
    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
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
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")

    # Forcing function
    f_expression = div(-grad(p_exact))
    f = Function(V).interpolate(f_expression)

    # Boundaries: Left (1), Right (2), Bottom(3), Top (4)
    vx = -2 * pi * cos(2 * pi * x) * sin(2 * pi * y)
    vy = -2 * pi * sin(2 * pi * x) * cos(2 * pi * y)
    p_boundaries = p_exact

    bc1 = DirichletBC(W[0], as_vector([vx, 0.0]), 1)
    bc2 = DirichletBC(W[0], as_vector([vx, 0.0]), 2)
    bc3 = DirichletBC(W[0], as_vector([0.0, vy]), 3)
    bc4 = DirichletBC(W[0], as_vector([0.0, vy]), 4)
    bcs = [bc1, bc2, bc3, bc4]

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p - q * div(u)) * dx
    L = -f * q * dx - p_boundaries * dot(v, n) * ds
    # Stabilizing terms
    a += -0.5 * inner((u + grad(p)), v + grad(q)) * dx
    # a += 0.5 * h * h * div(u) * div(v) * dx
    # a += 0.5 * h * h * inner(curl(u), curl(v)) * dx
    # L += 0.5 * h * h * f * div(v) * dx
    a += 0.5 * div(u) * div(v) * dx
    a += 0.5 * inner(curl(u), curl(v)) * dx
    L += 0.5 * f * div(v) * dx

    # Solving the system
    # The below code only uses SVD to estimate the condition number.
    # Caution is needed since it can be computationally expensive.
    solver_parameters = {
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'svd',
        'pc_svd_monitor': None,
        'ksp_monitor_singular_value': None,
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij'
    }
    problem = LinearVariationalProblem(a, L, solution, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.snes.ksp.setConvergenceHistory()
    solver.solve()

    sigma_h, u_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    u_h.rename('Pressure', 'label')

    return u_h, sigma_h


def solve_poisson_vms(num_elements_x, num_elements_y, degree=1, use_quads=False):

    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
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
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")

    # Forcing function
    f_expression = div(-grad(p_exact))
    f = Function(V).interpolate(f_expression)

    # Boundaries: Left (1), Right (2), Bottom(3), Top (4)
    vx = -2 * pi * cos(2 * pi * x) * sin(2 * pi * y)
    vy = -2 * pi * sin(2 * pi * x) * cos(2 * pi * y)
    p_boundaries = p_exact

    bc1 = DirichletBC(W[0], as_vector([vx, 0.0]), 1)
    bc2 = DirichletBC(W[0], as_vector([vx, 0.0]), 2)
    bc3 = DirichletBC(W[0], as_vector([0.0, vy]), 3)
    bc4 = DirichletBC(W[0], as_vector([0.0, vy]), 4)
    bcs = [bc1, bc2, bc3, bc4]

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p + q * div(u)) * dx
    L = f * q * dx - p_boundaries * dot(v, n) * ds
    # Stabilizing terms
    a += 0.5 * inner(u + grad(p), grad(q) - v) * dx
    # a += 0.5 * h * h * div(u) * div(v) * dx
    # a += 0.5 * h * h * inner(curl(u), curl(v)) * dx
    # L += 0.5 * h * h * f * div(v) * dx
    # a += 0.5 * div(u) * div(v) * dx
    # a += 0.5 * inner(curl(u), curl(v)) * dx
    # L += 0.5 * f * div(v) * dx

    # Solving the system
    # The below code only uses SVD to estimate the condition number.
    # Caution is needed since it can be computationally expensive.
    solver_parameters = {
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'svd',
        'pc_svd_monitor': None,
        'ksp_monitor_singular_value': None,
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij'
    }
    problem = LinearVariationalProblem(a, L, solution, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.snes.ksp.setConvergenceHistory()
    solver.solve()

    sigma_h, u_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    u_h.rename('Pressure', 'label')

    return u_h, sigma_h


def solve_poisson_mixed_RT(num_elements_x, num_elements_y, degree=1, use_quads=False):

    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
    if use_quads:
        hdiv_family = 'RTCF'
        pressure_family = 'DQ'
    else:
        hdiv_family = 'RT'
        pressure_family = 'DG'

    U = FunctionSpace(mesh, hdiv_family, degree + 1)
    V = FunctionSpace(mesh, pressure_family, degree)
    W = U * V

    # Trial and test functions
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    solution = Function(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")

    # Forcing function
    f_expression = div(-grad(p_exact))
    f = Function(V).interpolate(f_expression)

    # Boundaries: Left (1), Right (2), Bottom(3), Top (4)
    vx = -2 * pi * cos(2 * pi * x) * sin(2 * pi * y)
    vy = -2 * pi * sin(2 * pi * x) * cos(2 * pi * y)
    p_boundaries = p_exact

    bc1 = DirichletBC(W[0], as_vector([vx, 0.0]), 1)
    bc2 = DirichletBC(W[0], as_vector([vx, 0.0]), 2)
    bc3 = DirichletBC(W[0], as_vector([0.0, vy]), 3)
    bc4 = DirichletBC(W[0], as_vector([0.0, vy]), 4)
    bcs = [bc1, bc2, bc3, bc4]

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p + q * div(u)) * dx
    L = f * q * dx - p_boundaries * dot(v, n) * ds

    # Solving the system
    # The below code only uses SVD to estimate the condition number.
    # Caution is needed since it can be computationally expensive.
    solver_parameters = {
        'snes_type': 'ksponly',
        'ksp_type': 'preonly',
        'pc_type': 'svd',
        'pc_svd_monitor': None,
        'ksp_monitor_singular_value': None,
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij'
    }
    problem = LinearVariationalProblem(a, L, solution, bcs=bcs)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.snes.ksp.setConvergenceHistory()
    solver.solve()

    sigma_h, u_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    u_h.rename('Pressure', 'label')

    return u_h, sigma_h


out = StringIO()
with pipes(stdout=out, stderr=STDOUT):
    N = 10
    # u_h = solve_poisson_cg(N, N, degree=1, use_quads=True)
    # u_h, sigma_h = solve_poisson_ls(N, N, degree=1, use_quads=True)
    # u_h, sigma_h = solve_poisson_cgls(N, N, degree=1, use_quads=True)
    # u_h, sigma_h = solve_poisson_vms(N, N, degree=1, use_quads=False)
    u_h, sigma_h = solve_poisson_mixed_RT(N, N, degree=1, use_quads=False)

stdout = out.getvalue()
stdout_as_list = stdout.split()
stdout_by_line = stdout.split(sep='\n')

# Retrieving smallest singular value. This is necessary to eliminate too small singular values.
stdout_smallest_singular_values_full = stdout_by_line[1].split()
stdout_smallest_singular_values_as_str = stdout_smallest_singular_values_full[4:]
smallest_singular_values = [float(singular_value) for singular_value in
                            stdout_smallest_singular_values_as_str]

smallest_singular_values = np.array(smallest_singular_values)
zero_tol = 1e-10
smallest_singular_values = smallest_singular_values[smallest_singular_values > zero_tol]

# Retrieving largest singular value.
stdout_largest_singular_values_full = stdout_by_line[2].split()
stdout_largest_singular_values_as_str = stdout_largest_singular_values_full[5:]
largest_singular_values = [float(singular_value) for singular_value in
                           stdout_largest_singular_values_as_str]

largest_singular_values = np.array(largest_singular_values)

# Calculating the condition number
condition_number = largest_singular_values.max() / smallest_singular_values.min()

print(f"Condition number: {condition_number}")
print(f"Full output:\n{stdout}")

# Plotting solution field
tripcolor(u_h)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

quiver(sigma_h)
# plt.axis('off')
plt.show()
