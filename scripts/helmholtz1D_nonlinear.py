"""
[Adapted and refactored from Firedrake test suite by Diego Volpatto]

This demo program solves Helmholtz's equation
  - div D(u) grad u(x, y) + kappa u(x,y) = f(x, y)
with
   D(u) = 1 + alpha * u**2
   alpha = 0.1
   kappa = 1
on the unit square with source f given by
   f(x, y) = -8*pi^2*alpha*cos(2*pi*x)*cos(2*pi*y)^3*sin(2*pi*x)^2
             - 8*pi^2*alpha*cos(2*pi*x)^3*cos(2*pi*y)*sin(2*pi*y)^2
             + 8*pi^2*(alpha*cos(2*pi*x)^2*cos(2*pi*y)^2 + 1)
               *cos(2*pi*x)*cos(2*pi*y)
             + kappa*cos(2*pi*x)*cos(2*pi*y)
and the analytical solution
  u(x, y) = cos(x*2*pi)*cos(y*2*pi)
"""

from firedrake import *


def create_mesh_and_function_space(numel_x, numel_y, degree=1, quadrilateral=False):
    # Create mesh and define function space
    mesh = UnitSquareMesh(numel_x, numel_y, quadrilateral=quadrilateral)
    V = FunctionSpace(mesh, "CG", degree)
    return mesh, V


def helmholtz(
        V,
        kappa,
        alpha,
        parameters={},
        source=Expression(0.0)
):
    # Define variational problem
    u = Function(V)
    v = TestFunction(V)
    f = Function(V)
    D = 1 + alpha * u * u
    f.interpolate(source)
    a = (dot(grad(v), D * grad(u)) + kappa * v * u) * dx
    L = f * v * dx

    solve(a - L == 0, u, solver_parameters=parameters)

    return u


def plot_result(u):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import rc
        plt.rc('text', usetex=True)
        plt.rc('font', size=14)

        # Setting up the figure object
        plt.figure(dpi=300, figsize=(8, 6))
        plot(u)
        plt.show()
        return True
    except Exception as exception:
        print(exception)
        return False


def compute_errors(u, u_exact, space):
    f = Function(space)
    f.interpolate(u_exact)
    return sqrt(assemble(dot(u - f, u - f) * dx))


def run_convergence_test(
        u_exact,
        kappa,
        alpha,
        source,
        degree=1,
        exponent_min=4,
        exponent_max=8,
        quadrilateral=False,
        parameters={}
):
    import numpy as np
    from scipy.stats import linregress
    errors = np.array([])
    mesh_size = np.array([])
    for exponent in range(exponent_min, exponent_max):
        mesh, V = create_mesh_and_function_space(2.0 ** exponent, 2.0 ** exponent, degree=degree, quadrilateral=quadrilateral)
        u_h = helmholtz(V, kappa, alpha, source=source, parameters=parameters)
        mesh_size = np.append(mesh_size, 2 ** exponent)
        errors = np.append(errors, compute_errors(u_h, u_exact, V))

    errors_log = np.log2(errors)
    mesh_size_log = np.log2(mesh_size)
    error_slope, _, _, _, _ = linregress(mesh_size_log, errors_log)
    return np.abs(error_slope)


# Creating the mesh and function space
mesh, V = create_mesh_and_function_space(50, 50, quadrilateral=True)

# Helmholtz model parameters
alpha = 0.1
kappa = 1

# Source term
source = Expression(
    "-8*pi*pi*%(alpha)s*cos(2*pi*x[0])*cos(2*pi*x[1])\
    *cos(2*pi*x[1])*cos(2*pi*x[1])*sin(2*pi*x[0])*sin(2*pi*x[0])\
    - 8*pi*pi*%(alpha)s*cos(2*pi*x[0])*cos(2*pi*x[0])\
    *cos(2*pi*x[0])*cos(2*pi*x[1])*sin(2*pi*x[1])*sin(2*pi*x[1])\
    + 8*pi*pi*(%(alpha)s*cos(2*pi*x[0])*cos(2*pi*x[0])\
    *cos(2*pi*x[1])*cos(2*pi*x[1]) + 1)*cos(2*pi*x[0])*cos(2*pi*x[1])\
    + %(kappa)s*cos(2*pi*x[0])*cos(2*pi*x[1])"
    % {'alpha': alpha, 'kappa': kappa}
)

# Exact solution
sol_exact = Expression("cos(x[0]*2*pi)*cos(x[1]*2*pi)")

# Solver parameters
parameters = {
    'snes_type': 'newtonls',
    'ksp_type': 'preonly',
    'pc_type': 'lu'
}

convergence_rate = run_convergence_test(
    sol_exact,
    kappa,
    alpha,
    source,
    exponent_max=10,
    parameters=parameters,
    quadrilateral=True
)
print(convergence_rate)

sol = helmholtz(V, kappa, alpha, parameters=parameters, source=source)
plot_result(sol)
