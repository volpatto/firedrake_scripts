from io import StringIO

from firedrake import *
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


out = StringIO()
with pipes(stdout=out, stderr=STDOUT):
    N = 10
    u_h = solve_poisson_cg(N, N, use_quads=True)

stdout = out.getvalue()
stdout_as_list = stdout.split()

# print("\n*******************************************")
condition_number_index = stdout_as_list.index("number") + 1
condition_number_as_str = stdout_as_list[condition_number_index]
condition_number = float(condition_number_as_str[:-1])
print(f"Condition number: {condition_number}")

# Plotting solution field
tripcolor(u_h)
plt.xlabel("x")
plt.ylabel("y")
plt.show()
