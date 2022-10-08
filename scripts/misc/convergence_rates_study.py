import attr
from firedrake import *
from firedrake.petsc import PETSc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from scipy.stats import linregress
from slepc4py import SLEPc
import pandas as pd
from tqdm import tqdm
import os

matplotlib.use('Agg')


@attr.s
class ConditionNumberResult(object):
    form = attr.ib()
    assembled_form = attr.ib()
    condition_number = attr.ib()
    sparse_operator = attr.ib()
    number_of_dofs = attr.ib()
    nnz = attr.ib()
    is_operator_symmetric = attr.ib()
    bcs = attr.ib(default=list())    


def plot_matrix(assembled_form, **kwargs):
    """Provides a plot of a matrix."""
    fig, ax = plt.subplots(1, 1)

    petsc_mat = assembled_form.M.handle
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    Mnp = Mnp.toarray()

    # Eliminate rows and columns filled with zero entries
    Mnp = Mnp[~(Mnp==0).all(1)]
    idx = np.argwhere(np.all(Mnp[..., :] == 0, axis=0))
    Mnp = np.delete(Mnp, idx, axis=1)
    Am = np.ma.masked_values(Mnp, 0, rtol=1e-13)

    # Plot the matrix
    plot = ax.matshow(Am, **kwargs)

    # Remove axis ticks and values
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    return plot


def plot_matrix_mixed(assembled_form, **kwargs):
    """Provides a plot of a mixed matrix."""
    fig, ax = plt.subplots(1, 1)

    petsc_mat = assembled_form.M.handle

    f0_size = assembled_form.M[0, 0].handle.getSize()
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    Mnp = Mnp.toarray()

    # Eliminate rows and columns filled with zero entries
    Mnp = Mnp[~(Mnp==0).all(1)]
    idx = np.argwhere(np.all(Mnp[..., :] == 0, axis=0))
    Mnp = np.delete(Mnp, idx, axis=1)
    Am = np.ma.masked_values(Mnp, 0, rtol=1e-13)

    # Plot the matrix
    plot = ax.matshow(Am, **kwargs)

    # Remove axis ticks and values
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.axhline(y=f0_size[0] - 0.5, color="k")
    ax.axvline(x=f0_size[0] - 0.5, color="k")

    return plot


def plot_matrix_primal_hybrid_full(a_form, bcs=[], **kwargs):
    """Provides a plot of a full hybrid-mixed matrix."""
    fig, ax = plt.subplots(1, 1)

    assembled_form = assemble(a_form, bcs=bcs, mat_type="aij")
    petsc_mat = assembled_form.M.handle

    f0_size = assembled_form.M[0, 0].handle.getSize()

    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    Mnp = Mnp.toarray()

    # Eliminate rows and columns filled with zero entries
    Mnp = Mnp[~(Mnp==0).all(1)]
    idx = np.argwhere(np.all(Mnp[..., :] == 0, axis=0))
    Mnp = np.delete(Mnp, idx, axis=1)
    Am = np.ma.masked_values(Mnp, 0, rtol=1e-13)

    # Plot the matrix
    plot = ax.matshow(Am, **kwargs)

    # Remove axis ticks and values
    ax.tick_params(length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.axhline(y=f0_size[0] - 0.5, color="k")
    ax.axvline(x=f0_size[0] - 0.5, color="k")

    return plot


def plot_matrix_mixed_hybrid_full(a_form, bcs=[], **kwargs):
    """Provides a plot of a full hybrid-mixed matrix."""
    fig, ax = plt.subplots(1, 1)

    assembled_form = assemble(a_form, bcs=bcs, mat_type="aij")
    petsc_mat = assembled_form.M.handle

    f0_size = assembled_form.M[0, 0].handle.getSize()
    f1_size = assembled_form.M[1, 1].handle.getSize()

    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    Mnp = Mnp.toarray()

    # Eliminate rows and columns filled with zero entries
    Mnp = Mnp[~(Mnp==0).all(1)]
    idx = np.argwhere(np.all(Mnp[..., :] == 0, axis=0))
    Mnp = np.delete(Mnp, idx, axis=1)
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


def plot_matrix_hybrid_multiplier(a_form, trace_index=2, bcs=[], **kwargs):
    """Provides a plot of a condensed hybrid-mixed matrix for single scale problems."""
    fig, ax = plt.subplots(1, 1)

    _A = Tensor(a_form)
    A = _A.blocks
    idx = trace_index
    S = A[idx, idx] - A[idx, :idx] * A[:idx, :idx].inv * A[:idx, idx]
    Smat = assemble(S, bcs=bcs)

    petsc_mat = Smat.M.handle
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    Mnp = Mnp.toarray()

    # Eliminate rows and columns filled with zero entries
    Mnp = Mnp[~(Mnp==0).all(1)]
    idx = np.argwhere(np.all(Mnp[..., :] == 0, axis=0))
    Mnp = np.delete(Mnp, idx, axis=1)
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


def filter_real_part_in_array(array: np.ndarray, imag_threshold: float = 1e-5) -> np.ndarray:
    """Utility function to filter real part in a numpy array.

    :param array: 
        Array with real and complex numbers.

    :param imag_threshold:
        Threshold to cut off imaginary part in complex number.

    :return:
        Filtered array with only real numbers.
    """
    real_part_array = array.real[abs(array.imag) < 1e-5]
    return real_part_array


def calculate_condition_number(
    A, 
    num_of_factors, 
    backend: str = "scipy",
    use_sparse: bool = False,
    zero_tol: float = 1e-5
):
    backend = backend.lower()

    if backend == "scipy":
        size = A.getSize()
        Mnp = csr_matrix(A.getValuesCSR()[::-1], shape=size)
        Mnp.eliminate_zeros()

        if use_sparse:
            singular_values = svds(
                A=Mnp, 
                k=num_of_factors, 
                which="LM", 
                maxiter=5000, 
                return_singular_vectors=False, 
                solver="lobpcg"
            )
        else:
            M = Mnp.toarray()
            singular_values = svd(M, compute_uv=False, check_finite=False)

        singular_values = singular_values[singular_values > zero_tol]

        condition_number = singular_values.max() / singular_values.min()
    elif backend == "slepc":
        S = SLEPc.SVD()
        S.create()
        S.setOperator(A)
        S.setType(SLEPc.SVD.Type.LAPACK)
        S.setDimensions(nsv=num_of_factors)
        S.setTolerances(max_it=5000)
        S.setWhichSingularTriplets(SLEPc.SVD.Which.LARGEST)
        S.solve()

        num_converged_values = S.getConverged()
        singular_values_list = list()
        if num_converged_values > 0:
            for i in range(num_converged_values):
                singular_value = S.getValue(i)
                singular_values_list.append(singular_value)
        else:
            raise RuntimeError("SLEPc SVD has not converged.")

        singular_values = np.array(singular_values_list)

        singular_values = singular_values[singular_values > zero_tol]
        condition_number = singular_values.max() / singular_values.min()
    else:
        raise NotImplementedError("The required method for condition number estimation is currently unavailable.")

    return condition_number


def norm_trace(v, norm_type="L2", mesh=None):
    r"""Compute the norm of ``v``.

    :arg v: a ufl expression (:class:`~.ufl.classes.Expr`) to compute the norm of
    :arg norm_type: the type of norm to compute, see below for
         options.
    :arg mesh: an optional mesh on which to compute the norm
         (currently ignored).

    Available norm types are:

    - Lp :math:`||v||_{L^p} = (\int |v|^p)^{\frac{1}{p}} \mathrm{d}s`
    - H1 :math:`||v||_{H^1}^2 = \int (v, v) + (\nabla v, \nabla v) \mathrm{d}s`
    - Hdiv :math:`||v||_{H_\mathrm{div}}^2 = \int (v, v) + (\nabla\cdot v, \nabla \cdot v) \mathrm{d}s`
    - Hcurl :math:`||v||_{H_\mathrm{curl}}^2 = \int (v, v) + (\nabla \wedge v, \nabla \wedge v) \mathrm{d}s`

    """
    typ = norm_type.lower()
    p = 2
    if typ == 'l2':
        expr = inner(v, v)
    elif typ.startswith('l'):
        try:
            p = int(typ[1:])
            if p < 1:
                raise ValueError
        except ValueError:
            raise ValueError("Don't know how to interpret %s-norm" % norm_type)
        expr = inner(v, v)
    elif typ == 'h1':
        expr = inner(v, v) + inner(grad(v), grad(v))
    elif typ == "hdiv":
        expr = inner(v, v) + div(v)*div(v)
    elif typ == "hcurl":
        expr = inner(v, v) + inner(curl(v), curl(v))
    else:
        raise RuntimeError("Unknown norm type '%s'" % norm_type)

    return assemble((expr("+")**(p/2))*dS)**(1/p) + assemble((expr**(p/2))*ds)**(1/p)


def errornorm_trace(u, uh, norm_type="L2", degree_rise=None, mesh=None):
    """Compute the error :math:`e = u - u_h` in the specified norm.

    :arg u: a :class:`.Function` or UFL expression containing an "exact" solution
    :arg uh: a :class:`.Function` containing the approximate solution
    :arg norm_type: the type of norm to compute, see :func:`.norm` for
         details of supported norm types.
    :arg degree_rise: ignored.
    :arg mesh: an optional mesh on which to compute the error norm
         (currently ignored).
    """
    urank = len(u.ufl_shape)
    uhrank = len(uh.ufl_shape)

    if urank != uhrank:
        raise RuntimeError("Mismatching rank between u and uh")

    if not isinstance(uh, function.Function):
        raise ValueError("uh should be a Function, is a %r", type(uh))

    if isinstance(u, function.Function):
        degree_u = u.function_space().ufl_element().degree()
        degree_uh = uh.function_space().ufl_element().degree()
        if degree_uh > degree_u:
            warning("Degree of exact solution less than approximation degree")

    return norm_trace(u - uh, norm_type=norm_type, mesh=mesh)


def calculate_exact_solution(mesh, pressure_family, velocity_family, pressure_degree, velocity_degree, is_hdiv_space=False):
    if is_hdiv_space:
        U = FunctionSpace(mesh, velocity_family, velocity_degree)
    else:
        U = VectorFunctionSpace(mesh, velocity_family, velocity_degree)
    V = FunctionSpace(mesh, pressure_family, pressure_degree)

    x, y = SpatialCoordinate(mesh)

    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).project(p_exact)
    exact_solution.rename("Exact pressure", "label")
    exact_velocity = Function(U, name='Exact velocity')
    exact_velocity.project(-grad(p_exact))
    
    return exact_solution, exact_velocity


def calculate_exact_solution_with_trace(mesh, pressure_family, velocity_family, pressure_degree, velocity_degree):
    U = VectorFunctionSpace(mesh, velocity_family, velocity_degree)
    V = FunctionSpace(mesh, pressure_family, pressure_degree)

    x, y = SpatialCoordinate(mesh)

    p_exact = sin(2 * pi * x) * sin(2 * pi * y)  # original
    # p_exact = sin(0.5 * pi * x) * sin(0.5 * pi * y)
    # p_exact = x * x * x - 3 * x * y * y
    # p_exact = - (x * x / 2 - x * x * x / 3) * (y * y / 2 - y * y * y / 3)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")
    exact_velocity = Function(U, name='Exact velocity')
    exact_velocity.project(-grad(p_exact))
    
    return exact_solution, exact_velocity, p_exact


def solve_poisson_cg(mesh, degree=1, use_quads=False):
    # Function space declaration
    pressure_family = 'CG'
    velocity_family = 'CG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)

    # Trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Exact solution
    exact_solution, sigma_e = calculate_exact_solution(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Dirichlet BCs
    bcs = DirichletBC(V, project(exact_solution, V), "on_boundary")

    # Variational form
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx

    # Solving the problem
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solution = Function(V)
    problem = LinearVariationalProblem(a, L, solution, bcs=bcs, constant_jacobian=False)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    p_h = solution
    sigma_h = project(-grad(p_h), U)

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_ls(mesh, degree=1):
    # Function space declaration
    pressure_family = 'CG'
    velocity_family = 'CG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    W = U * V

    # Trial and test functions
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Mesh entities
    h = CellDiameter(mesh)

    # Exact solution
    exact_solution, sigma_e = calculate_exact_solution(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Dirichlet BCs
    bcs = DirichletBC(W[0], project(sigma_e, U), "on_boundary")

    # Stabilization parameters
    delta_1 = Constant(1)
    delta_2 = Constant(1)
    delta_3 = Constant(1)

    # Least-squares terms
    a = delta_1 * inner(u + grad(p), v + grad(q)) * dx
    a += delta_2 * div(u) * div(v) * dx
    a += delta_3 * inner(curl(u), curl(v)) * dx
    L = delta_2 * f * div(v) * dx

    # Weakly imposed BC
    huge_number = 1e12
    nitsche_penalty = Constant(huge_number)
    p_e = exact_solution
    a += (nitsche_penalty / h) * p * q * ds
    L += (nitsche_penalty / h) * p_e * q * ds

    # Solving the problem
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solution = Function(W)
    problem = LinearVariationalProblem(a, L, solution, bcs=bcs, constant_jacobian=False)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h = solution.split()

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_cgls(mesh, degree=1):
    # Function space declaration
    pressure_family = 'CG'
    velocity_family = 'CG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    W = U * V

    # Trial and test functions
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e = calculate_exact_solution(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Dirichlet BCs
    bc1 = DirichletBC(W[0], project(sigma_e, U), "on_boundary")
    bc2 = DirichletBC(W[1], project(exact_solution, V), "on_boundary")
    bcs = [bc1, bc2]

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p - q * div(u)) * dx
    L = -f * q * dx - exact_solution * dot(v, n) * ds
    # Stabilizing terms
    a += -0.5 * inner((u + grad(p)), v + grad(q)) * dx
    # a += 0.5 * h * h * div(u) * div(v) * dx
    # a += 0.5 * h * h * inner(curl(u), curl(v)) * dx
    # L += 0.5 * h * h * f * div(v) * dx
    a += 0.5 * div(u) * div(v) * dx
    a += 0.5 * inner(curl(u), curl(v)) * dx
    L += 0.5 * f * div(v) * dx

    # Solving the problem
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solution = Function(W)
    problem = LinearVariationalProblem(a, L, solution, bcs=bcs, constant_jacobian=False)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h = solution.split()

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_vms(mesh, degree=1):
    # Function space declaration
    pressure_family = 'CG'
    velocity_family = 'CG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    W = U * V

    # Trial and test functions
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")
    sigma_e = Function(U, name='Exact velocity')
    sigma_e.project(-grad(p_exact))

    # Dirichlet BCs
    bcs = DirichletBC(W[0], sigma_e, "on_boundary")

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p + q * div(u)) * dx
    # Stabilizing terms
    a += 0.5 * inner(u + grad(p), grad(q) - v) * dx
    # a += 0.5 * h * h * div(u) * div(v) * dx
    # a += 0.5 * h * h * inner(curl(u), curl(v)) * dx
    # L += 0.5 * h * h * f * div(v) * dx
    # a += 0.5 * div(u) * div(v) * dx
    # a += 0.5 * inner(curl(u), curl(v)) * dx
    # L += 0.5 * f * div(v) * dx

    A = assemble(a, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()

    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = W.dim()

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=A,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric
    )
    return result


def solve_poisson_mixed_RT(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    if use_quads:
        hdiv_family = 'RTCF'
        pressure_family = 'DQ'
    else:
        hdiv_family = 'RT'
        pressure_family = 'DG'

    U = FunctionSpace(mesh, hdiv_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree - 1)
    W = U * V

    # Trial and test functions
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)

    # Exact solution
    exact_solution, sigma_e = calculate_exact_solution(
        mesh, 
        pressure_family, 
        hdiv_family, 
        degree + 3, 
        degree + 3,
        is_hdiv_space=True
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Dirichlet BCs
    # bcs = DirichletBC(W[0], sigma_e, "on_boundary")

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p + q * div(u)) * dx
    L = f * q * dx - dot(v, n) * exact_solution * ds

    # Solving the problem
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solution = Function(W)
    problem = LinearVariationalProblem(a, L, solution, bcs=[], constant_jacobian=False)
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h = solution.split()

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_dgls(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    W = U * V

    # Trial and test functions
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")
    sigma_e = Function(U, name='Exact velocity')
    sigma_e.project(-grad(p_exact))

    # Dirichlet BCs
    # bcs = DirichletBC(W[0], sigma_e, "on_boundary", method="geometric")

    # Average cell size and mesh dependent stabilization
    h_avg = (h("+") + h("-")) / 2.0

    # Jump stabilizing parameters based on Badia-Codina stabilized dG method
    L0 = 1
    eta_p = L0 * h  # method B in the Badia-Codina paper
    # eta_p = 1
    # eta_p = L0 * L0  # method D in the Badia-Codina paper
    eta_u = h / L0  # method B in the Badia-Codina paper
    # eta_u = 1

    # Nitsche's penalizing term
    beta_0 = Constant(1.0)
    beta = beta_0 / h

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p - q * div(u)) * dx
    # DG terms
    a += jump(v, n) * avg(p) * dS - avg(q) * jump(u, n) * dS
    # Edge stabilizing terms
    # ** Badia-Codina based
    a += (avg(eta_p) / h_avg) * (jump(u, n) * jump(v, n)) * dS
    a += (avg(eta_u) / h_avg) * dot(jump(p, n), jump(q, n)) * dS
    # ** Mesh independent terms
    # a += jump(u, n) * jump(v, n) * dS
    # a += dot(jump(p, n), jump(q, n)) * dS
    # Volumetric stabilizing terms
    # a += 0.5 * h * h * div(u) * div(v) * dx
    # a += 0.5 * h * h * inner(curl(u), curl(v)) * dx
    # L += 0.5 * h * h * f * div(v) * dx
    # a += -0.5 * inner(u + grad(p), v + grad(q)) * dx
    # a += 0.5 * div(u) * div(v) * dx
    # a += 0.5 * inner(curl(u), curl(v)) * dx
    # ** Badia-Codina based
    a += -eta_u * inner(u + grad(p), v + grad(q)) * dx
    a += eta_p * div(u) * div(v) * dx
    a += eta_p * inner(curl(u), curl(v)) * dx
    # Weakly imposed boundary conditions
    a += dot(v, n) * p * ds - q * dot(u, n) * ds
    a += beta * p * q * ds  # may decrease convergente rates
    # ** The terms below are based on ASGS Badia-Codina (2010), it is not a classical Nitsche's method
    a += (eta_p / h) * dot(u, n) * dot(v, n) * ds
    a += (eta_u / h) * dot(p * n, q * n) * ds

    A = assemble(a, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = W.dim()

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=A,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric
    )
    return result


def solve_poisson_dvms(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    W = U * V

    # Trial and test functions
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")
    sigma_e = Function(U, name='Exact velocity')
    sigma_e.project(-grad(p_exact))

    # Dirichlet BCs
    # bcs = DirichletBC(W[0], sigma_e, "on_boundary", method="geometric")

    # Average cell size and mesh dependent stabilization
    h_avg = (h("+") + h("-")) / 2.0

    # Jump stabilizing parameters based on Badia-Codina stabilized dG method
    L0 = 1
    eta_p = L0 * h  # method B in the Badia-Codina paper
    # eta_p = L0 * L0  # method D in the Badia-Codina paper
    eta_u = h / L0  # method B in the Badia-Codina paper

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p + q * div(u)) * dx
    # DG terms
    a += jump(v, n) * avg(p) * dS - avg(q) * jump(u, n) * dS
    # Edge stabilizing terms
    # ** Badia-Codina based
    a += (avg(eta_p) / h_avg) * (jump(u, n) * jump(v, n)) * dS
    a += (avg(eta_u) / h_avg) * dot(jump(p, n), jump(q, n)) * dS
    # ** Mesh independent (original)
    # a += jump(u, n) * jump(v, n) * dS  # not considered in the original paper
    # a += dot(jump(p, n), jump(q, n)) * dS
    # Volumetric stabilizing terms
    # a += 0.5 * inner(u + grad(p), grad(q) - v) * dx
    # a += 0.5 * h * h * div(u) * div(v) * dx
    # a += 0.5 * h * h * inner(curl(u), curl(v)) * dx
    # L += 0.5 * h * h * f * div(v) * dx
    # a += 0.5 * div(u) * div(v) * dx
    # a += 0.5 * inner(curl(u), curl(v)) * dx
    # L += 0.5 * f * div(v) * dx
    # ** Badia-Codina based
    a += eta_u * inner(u + grad(p), grad(q) - v) * dx
    a += eta_p * div(u) * div(v) * dx
    # Weakly imposed boundary conditions
    a += dot(v, n) * p * ds - q * dot(u, n) * ds
    # ** The terms below are based on ASGS Badia-Codina (2010), it is not a classical Nitsche's method
    a += (eta_p / h) * dot(u, n) * dot(v, n) * ds
    a += (eta_u / h) * dot(p * n, q * n) * ds  # may decrease convergente rates
    # ** Classical Nitsche
    # a += beta * p * q * ds  # may decrease convergente rates (Nitsche)

    A = assemble(a, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = W.dim()

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=A,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric
    )
    return result


def solve_poisson_sipg(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)

    # Trial and test functions
    p = TrialFunction(V)
    q = TestFunction(V)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e = calculate_exact_solution(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Edge stabilizing parameter
    beta0 = Constant(1e2)
    beta = beta0 / h

    # Symmetry term. Choose if the method is SIPG (-1) or NIPG (1)
    s = Constant(-1)

    # Classical volumetric terms
    a = inner(grad(p), grad(q)) * dx
    L = f * q * dx
    # DG edge terms
    a += s * dot(jump(p, n), avg(grad(q))) * dS - dot(avg(grad(p)), jump(q, n)) * dS
    # Edge stabilizing terms
    a += avg(beta) * dot(jump(p, n), jump(q, n)) * dS
    # Weak boundary conditions
    a += s * dot(p * n, grad(q)) * ds - dot(grad(p), q * n) * ds
    a += beta * p * q * ds
    L += beta * exact_solution * q * ds

    # Solving the system
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solution = Function(V)
    problem = LinearVariationalProblem(a, L, solution, bcs=[])
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    p_h = solution
    sigma_h = project(-grad(p_h), U)

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_dls_primal(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)

    # Trial and test functions
    p = TrialFunction(V)
    q = TestFunction(V)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e = calculate_exact_solution(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Stabilizing parameter
    # delta_base = h * h
    element_size_factor = h
    penalty_constant = 1e0
    penalty_constant_ip = 1e2
    delta_base = Constant(penalty_constant * degree * degree)
    # delta_base = Constant(penalty_constant)
    # delta_base = Constant(1e0 * degree * degree)
    enable_dg_ip = Constant(0)  # enable (1) or disable (0)
    delta_0 = delta_base / delta_base * enable_dg_ip
    delta_1 = Constant(1)
    # delta_1 = h * h
    # delta_2 = delta_base / h
    # Testar esses valores, Abimael acha que é ao cubo. Testar combinações
    # delta_2 = delta_base / h / h / h / h * Constant(1)
    delta_2 = delta_base / h / h * Constant(1)
    # delta_2 = delta_base / h * Constant(1)  # DG-IP
    # delta_3 = 1 / delta_base / h / h
    # delta_3 = 1 / delta_base * element_size_factor * Constant(1)
    delta_3 = Constant(1)
    # delta_3 = Constant(1) * delta_base
    # delta_3 = delta_2
    
    # Stabilizing parameter (for PhD results)
    # penalty_constant = 1e0
    # delta_base = Constant(penalty_constant * degree * degree)
    # enable_dg_ip = Constant(0)  # enable (1) or disable (0)
    # delta_0 = delta_base / delta_base * enable_dg_ip
    # delta_1 = Constant(1)
    # delta_2 = delta_base / h / h * Constant(1)
    # delta_3 = Constant(1)
    
    # Stabilizing parameter (testing)
    penalty_constant = 1e2
    penalty_constant_ip = 0e0
    # delta_base = Constant(penalty_constant * degree * degree)
    delta_base = Constant(penalty_constant)
    enable_dg_ip = Constant(0)  # enable (1) or disable (0)
    delta_0 = delta_base / delta_base * enable_dg_ip
    delta_1 = Constant(1)
    delta_2 = delta_base / h / h / h / h * Constant(1) / delta_base
    delta_3 = Constant(1) / delta_base / h / h * delta_base

    # Flux variables
    u = -grad(p)
    v = -grad(q)

    # Residual definition
    Lp = div(u)
    Lq = div(v)

    # Classical DG-IP term
    a = delta_0 * dot(grad(p), grad(q)) * dx
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
    a += delta_2 * (p - exact_solution) * q * ds
    a += avg(delta_3) * jump(u, n=n) * jump(v, n=n) * dS
    # Flux should not be imposed at Dirichlet condition boundaries
    # a += delta_3 * dot(u, n) * dot(v, n) * ds
    # L += delta_3 * dot(sigma_e, n) * dot(v, n) * ds

    # DG-IP Weak boundary conditions (not required, already imposed by LS terms)
    beta0 = Constant(enable_dg_ip * penalty_constant_ip * degree * degree)
    beta = beta0 / h  # Nitsche term
    a += s * delta_0 * dot(p * n, v) * ds - delta_0 * dot(u, q * n) * ds
    a += delta_0 * beta * p * q * ds
    L += delta_0 * beta * exact_solution * q * ds

    # Ensuring that the formulation is properly decomposed in LHS and RHS
    F = a - L
    a_form = lhs(F)
    L_form = rhs(F)

    # Solving the system
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solution = Function(V)
    problem = LinearVariationalProblem(a_form, L_form, solution, bcs=[])
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    p_h = solution
    sigma_h = project(-grad(p_h), U)

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_cls_primal(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'CG' if use_quads else 'CG'
    velocity_family = 'CG' if use_quads else 'CG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)

    # Trial and test functions
    p = TrialFunction(V)
    q = TestFunction(V)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e = calculate_exact_solution(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))
    
    # Stabilizing parameter (for PhD results)
    element_size_factor = h
    beta0 = Constant(1e5)
    beta = beta0 / h
    penalty_constant = 1e0
    delta_base = Constant(penalty_constant * degree * degree)
    enable_dg_ip = Constant(0)  # enable (1) or disable (0)
    delta_0 = delta_base / delta_base * enable_dg_ip
    delta_1 = Constant(1)
    delta_2 = delta_base / h / h * Constant(1)
    delta_3 = Constant(1) * beta * h
    # delta_3 = Constant(1e-2)

    # Flux variables
    u = -grad(p)
    v = -grad(q)

    # Residual definition
    Lp = div(u)
    Lq = div(v)

    # Classical DG-IP term
    a = delta_0 * dot(grad(p), grad(q)) * dx
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
    a += delta_2 * (p - exact_solution) * q * ds
    a += avg(delta_3) * jump(u, n=n) * jump(v, n=n) * dS
    # Flux should not be imposed at Dirichlet condition boundaries
    # a += delta_3 * dot(u, n) * dot(v, n) * ds
    # L += delta_3 * dot(sigma_e, n) * dot(v, n) * ds

    # DG-IP Weak boundary conditions (not required, already imposed by LS terms)
    # beta0 = Constant(enable_dg_ip * penalty_constant_ip * degree * degree)
    # beta = beta0 / h  # Nitsche term
    # a += s * delta_0 * dot(p * n, v) * ds - delta_0 * dot(u, q * n) * ds
    # a += delta_0 * beta * p * q * ds
    # L += delta_0 * beta * exact_solution * q * ds

    # Ensuring that the formulation is properly decomposed in LHS and RHS
    F = a - L
    a_form = lhs(F)
    L_form = rhs(F)

    # Solving the system
    solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    solution = Function(V)
    problem = LinearVariationalProblem(a_form, L_form, solution, bcs=[])
    solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
    solver.solve()

    # Retrieving the solution
    p_h = solution
    sigma_h = project(-grad(p_h), U)

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_dls(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    W = U * V

    # Trial and test functions
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")
    sigma_e = Function(U, name='Exact velocity')
    sigma_e.project(-grad(p_exact))

    # Dirichlet BCs
    # bcs = DirichletBC(W[0], sigma_e, "on_boundary", method="geometric")

    # Average cell size and mesh dependent stabilization
    h_avg = (h("+") + h("-")) / 2.0

    # Jump stabilizing parameters based on Badia-Codina stabilized dG method
    # L0 = 1
    # eta_p = L0 * h_avg  # method B in the Badia-Codina paper
    eta_p = 1
    # eta_p = L0 * L0  # method D in the Badia-Codina paper
    # eta_u = h_avg / L0  # method B in the Badia-Codina paper
    eta_u = 1
    # eta_u_bc = h / L0  # method B in the Badia-Codina paper
    eta_u_bc = 1

    # Least-Squares weights
    delta = Constant(1.0)
    # delta = h
    delta_0 = delta
    delta_1 = delta
    delta_2 = delta
    delta_3 = 1 / h
    delta_4 = 1 / h

    # Least-squares terms
    a = delta_0 * inner(u + grad(p), v + grad(q)) * dx
    a += delta_1 * div(u) * div(v) * dx
    a += delta_2 * inner(curl(u), curl(v)) * dx
    # Edge stabilizing terms
    # ** Badia-Codina based (better results) **
    a += eta_u * avg(delta_3) * (jump(u, n) * jump(v, n)) * dS
    a += eta_p * avg(delta_4) * dot(jump(p, n), jump(q, n)) * dS
    a += eta_u_bc * delta_3 * p * q * ds  # may decrease convergente rates
    a += eta_u_bc * delta_4 * dot(u, n) * dot(v, n) * ds
    # ** Mesh independent **
    # a += jump(u, n) * jump(v, n) * dS
    # a += dot(jump(p, n), jump(q, n)) * dS
    # a += p * q * ds

    A = assemble(a, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-12)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = W.dim()

    num_of_factors = int(number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors)

    result = ConditionNumberResult(
        form=a,
        assembled_form=A,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric
    )
    return result


def solve_poisson_sdhm(
    mesh, 
    degree=1,
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    trace_family = "HDiv Trace"
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        T = FunctionSpace(mesh, trace_family, degree)
    W = U * V * T

    # Trial and test functions
    solution = Function(W)
    u, p, lambda_h = split(solution)
    # u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # BCs
    u_projected = sigma_e
    p_boundaries = exact_solution
    bcs = DirichletBC(W.sub(2), exact_trace, "on_boundary")

    # Hybridization parameter
    beta_0 = Constant(0.0e0)
    beta = beta_0 / h
    # beta = beta_0

    # Stabilization parameters
    delta_0 = Constant(-1)
    delta_1 = Constant(-0.5)  #* h * h
    delta_2 = Constant(0.5)  #* h * h
    delta_3 = Constant(0.5)  #* h * h

    # # Mixed classical terms
    # a = (dot(u, v) - div(v) * p + delta_0 * q * div(u)) * dx
    # L = delta_0 * f * q * dx
    # # Stabilizing terms
    # a += delta_1 * inner(u + grad(p), v + grad(q)) * dx
    # a += delta_2 * div(u) * div(v) * dx
    # a += delta_3 * inner(curl(u), curl(v)) * dx
    # L += delta_2 * f * div(v) * dx
    # # Hybridization terms
    # a += lambda_h("+") * dot(v, n)("+") * dS + mu_h("+") * dot(u, n)("+") * dS
    # a += beta("+") * (lambda_h("+") - p("+")) * (mu_h("+") - q("+")) * dS
    # # Weakly imposed BC
    # a += (p_boundaries * dot(v, n) + mu_h * (dot(u, n) - dot(u_projected, n))) * ds
    # a += beta * (lambda_h - p_boundaries) * mu_h * ds

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # HDG classical form
    a = (dot(u, v) - div(v) * p) * dx + lambda_h("+") * jump(v, n) * dS
    a += delta_0 * div(u) * q * dx
    L = delta_0 * f * q * dx

    # Least-squares terms
    a += delta_1 * inner(u + grad(p), v + grad(q)) * dx
    a += delta_2 * div(u) * div(v) * dx
    a += delta_3 * inner(curl(u), curl(v)) * dx
    L += delta_2 * f * div(v) * dx

    # Transmission condition
    a += jump(u_hat, n) * mu_h("+") * dS

    # Nitsche's term to transmission condition edge stabilization
    a += -beta('+') * (p('+') - lambda_h('+')) * q('+') * dS

    # Weakly imposed BC
    # a += lambda_h * dot(v, n) * ds  # required term
    L += -exact_solution * dot(v, n) * ds  # required as the above, but just one of them should be used (works for continuous multiplier)
    a += -beta * p * q * ds  # required term... note that u (the unknown) is used
    L += -beta * exact_solution * q * ds  # Required, this one is paired with the above term
    a += lambda_h * mu_h * ds  # Classical required term
    L += exact_solution * mu_h * ds  # Pair for the above classical required term

    F = a - L

    params = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "pmat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        # Use the static condensation PC for hybridized problems
        # and use a direct solve on the reduced system for lambda_h
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0, 1",
        "condensed_field": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        },
    }
    # problem = NonlinearVariationalProblem(F, solution, bcs=bcs)
    problem = NonlinearVariationalProblem(F, solution)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h, lambda_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")
    # interpolate(exact_trace, T)
    # p_error_L2 = errornorm_trace(interpolate(exact_trace, T), lambda_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_hdg(
    mesh, 
    degree=1,
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    trace_family = "HDiv Trace"
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        T = FunctionSpace(mesh, trace_family, degree)
    W = U * V * T

    # Trial and test functions
    solution = Function(W)
    u, p, lambda_h = split(solution)
    # u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Dirichlet BCs
    bcs = DirichletBC(W.sub(2), exact_trace, "on_boundary")

    # Hybridization parameter
    beta_0 = Constant(1.0e0)
    beta = beta_0 / h
    # beta = beta_0

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # HDG classical form
    a = (dot(u, v) - div(v) * p) * dx + lambda_h("+") * jump(v, n) * dS
    a += -dot(u, grad(q)) * dx + jump(u_hat, n) * q("+") * dS
    L = f * q * dx
    # Transmission condition
    a += jump(u_hat, n) * mu_h("+") * dS
    # Weakly imposed BC
    a += lambda_h * dot(v, n) * ds
    a += dot(u_hat, n) * q * ds

    F = a - L

    params = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "pmat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        # Use the static condensation PC for hybridized problems
        # and use a direct solve on the reduced system for lambda_h
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0, 1",
        "condensed_field": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"
        },
    }
    problem = NonlinearVariationalProblem(F, solution, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h, lambda_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")
    # p_error_L2 = errornorm_trace(interpolate(exact_trace, T), lambda_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_cgh(
    mesh, 
    degree=1, 
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    trace_family = "HDiv Trace"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        T = FunctionSpace(mesh, trace_family, degree)
    W = V * T

    # Trial and test functions
    solution = Function(W)
    p, lambda_h = split(solution)
    # u, p, lambda_h = TrialFunctions(W)
    q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Dirichlet BCs
    bc_multiplier = DirichletBC(W.sub(1), exact_trace, "on_boundary")

    # Hybridization parameter
    beta_0 = Constant(1.0e0)
    beta = beta_0 / h
    # beta = beta_0

    # Numerical flux trace
    u = -grad(p)
    u_hat = u + beta * (p - lambda_h) * n

    # HDG classical form
    a = -dot(u, grad(q)) * dx + jump(u_hat, n) * q("+") * dS
    # a = div(u) * q * dx
    L = f * q * dx
    # Transmission condition
    a += jump(u_hat, n) * mu_h("+") * dS
    # Weakly imposed BC
    a += dot(u_hat, n) * q * ds
    a += mu_h * (lambda_h - exact_trace) * ds

    # a += (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
    # a += (p - lambda_h) * (q - mu_h) * ds

    F = a - L

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
            "pc_factor_mat_solver_type": "mumps",
            # "mat_mumps_icntl_4": "2",
        },
    }
    # problem = NonlinearVariationalProblem(F, solution, bcs=bcs)
    problem = NonlinearVariationalProblem(F, solution)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    p_h, lambda_h = solution.split()
    sigma_h = Function(U, name='Velocity')
    sigma_h.project(-grad(p_h))
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_ldgc(
    mesh, 
    degree=1, 
    is_multiplier_continuous=True
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        trace_family = "HDiv Trace"
        T = FunctionSpace(mesh, trace_family, degree)
    W = V * T

    # Trial and test functions
    solution = Function(W)
    p, lambda_h = split(solution)
    # u, p, lambda_h = TrialFunctions(W)
    q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # Dirichlet BCs
    bc_multiplier = DirichletBC(W.sub(1), exact_trace, "on_boundary")

    # Hybridization parameter
    # beta = Constant(6.0) if degree == 1 else Constant(15)
    beta = Constant(8.0 * degree * degree) / h

    # Numerical flux trace
    u = -grad(p)
    u_hat = u + beta * (p - lambda_h) * n

    # Symmetry parameter: s = -1 (symmetric), s = 1 (unsymmetric), and s = 0 (no-symmetrization)
    s = Constant(-1)

    # Classical Galerkin form
    a = -dot(u, grad(q)) * dx + jump(u_hat, n) * q("+") * dS
    # a = div(u) * q * dx
    L = f * q * dx
    # Transmission condition
    a += -jump(u_hat, n) * mu_h("+") * dS
    # Symmetrization
    a += s * jump(grad(q), n) * (p('+') - lambda_h("+")) * dS
    a += s * dot(grad(q), n) * (p - exact_solution) * ds
    # Weakly imposed BC
    # a += dot(u_hat, n) * q * ds
    a += dot(u, n) * q * ds	+ beta * (p - exact_solution) * q * ds  # expand u_hat product in ds
    a += mu_h * (lambda_h - exact_solution) * ds
    F = a - L

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
            "pc_factor_mat_solver_type": "mumps",
            # "mat_mumps_icntl_4": "2",
        },
    }
    # problem = NonlinearVariationalProblem(F, solution, bcs=bc_multiplier)
    problem = NonlinearVariationalProblem(F, solution)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    p_h, lambda_h = solution.split()
    sigma_h = Function(U, name='Velocity')
    sigma_h.project(-grad(p_h))
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_lsh(
    mesh, 
    degree=1, 
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        trace_family = "HDiv Trace"
        T = FunctionSpace(mesh, trace_family, degree)
    W = U * V * T

    # Trial and test functions
    solution = Function(W)
    u, p, lambda_h = split(solution)
    # u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))
    f = Function(V).interpolate(f)

    # BCs
    p_exact = Constant(0)
    # bcs = DirichletBC(W.sub(2), exact_trace, "on_boundary")
    bcs = DirichletBC(W.sub(2), p_exact, "on_boundary")

    # Hybridization parameter
    beta_0 = Constant(1.0e0)
    # beta = beta_0 / h
    beta = beta_0
    beta_avg = beta_0 / h("+")

    # Stabilizing parameter
    # delta_0 = Constant(1)
    # delta_1 = Constant(1)
    # delta_2 = Constant(1)
    # delta_3 = Constant(1)
    # delta_4 = Constant(1)
    # delta_5 = Constant(1)
    delta = h * h
    # delta = Constant(1)
    LARGE_NUMBER = Constant(1e0)
    # delta = 1 / h
    delta_0 = delta
    # delta_1 = delta
    delta_1 = Constant(1) * delta
    delta_2 = delta
    delta_3 = delta
    delta_4 = delta  #/ h
    delta_5 = delta * Constant(0)  #/ h
    # delta_5 = LARGE_NUMBER / h

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n
    v_hat = v + beta * (q - mu_h) * n

    # Flux least-squares
    a = (
        (inner(u, v) - q * div(u) - p * div(v) + inner(grad(p), grad(q)))
        * delta_1
        * dx
    )
    # a = (
    #     (inner(u, v) - q * div(u) + dot(v, grad(p)) + inner(grad(p), grad(q)))
    #     * delta_1
    #     * dx
    # )
    # These terms below are unsymmetric
    a += delta_1("+") * jump(u_hat, n=n) * q("+") * dS
    a += delta_1 * dot(u_hat, n) * q * ds
    # a += delta_1 * dot(u, n) * q * ds
    # L = -delta_1 * dot(u_projected, n) * q * ds
    a += delta_1("+") * lambda_h("+") * jump(v, n=n) * dS
    a += delta_1 * lambda_h * dot(v, n) * ds

    # Flux Least-squares as in DG
    # a = delta_0 * inner(u + grad(p), v + grad(q)) * dx

    # Classical mixed Darcy eq. first-order terms as stabilizing terms
    # a += delta_1 * (dot(u, v) - div(v) * p) * dx
    # a += delta_1("+") * lambda_h("+") * jump(v, n=n) * dS
    # a += delta_1 * lambda_h * dot(v, n) * ds

    # Mass balance least-square
    a += delta_2 * div(u) * div(v) * dx
    L = delta_2 * f * div(v) * dx

    # Irrotational least-squares
    a += delta_3 * inner(curl(u), curl(v)) * dx

    # Hybridization terms
    a += mu_h("+") * jump(u_hat, n=n) * dS
    a += mu_h * (lambda_h - p_exact) * ds
    # a += mu_h * dot(u_hat, n) * ds
    # L += mu_h * dot(sigma_e, n) * ds
    a += delta_4("+") * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
    a += delta_4 * (p - lambda_h) * (q - mu_h) * ds
    # a += delta_4 * (exact_solution - lambda_h) * (q - mu_h) * ds
    # Alternative primal
    # a += delta_4("+") * (lambda_h("+") - p("+")) * (mu_h("+") - q("+")) * dS
    # a += delta_4 * (lambda_h - p) * (mu_h - q) * ds
    # Flux
    a += delta_5("+") * (dot(u, n)("+") - dot(u_hat, n)("+")) * (dot(v, n)("+") - dot(v_hat, n)("+")) * dS
    a += delta_5 * (dot(u, n) - dot(u_hat, n)) * (dot(v, n) - dot(v_hat, n)) * ds
    # Alternative
    # a += delta_5("+") * (dot(u_hat, n)("+") - dot(u, n)("+")) * (dot(v_hat, n)("+") - dot(v, n)("+")) * dS
    # a += delta_5 * (dot(u_hat, n) - dot(u, n)) * (dot(v_hat, n) - dot(v, n)) * ds

    # Weakly imposed BC from hybridization
    # a += mu_h * (lambda_h - exact_trace) * ds
    # a += mu_h * lambda_h * ds
    # ###
    # a += (
    #     delta_4 * (mu_h - q) * (lambda_h - exact_solution) * ds
    # )  # maybe this is not a good way to impose BC, but this necessary
    # a += (
    #     delta_4 * (q - mu_h) * (exact_solution - lambda_h) * ds
    # )  # maybe this is not a good way to impose BC, but this necessary
    L += delta_1 * exact_solution * dot(v, n) * ds  # study if this is a good BC imposition

    F = a - L

    params = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "pmat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        # Use the static condensation PC for hybridized problems
        # and use a direct solve on the reduced system for lambda_h
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0, 1",
        "condensed_field": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }
    # problem = NonlinearVariationalProblem(F, solution, bcs=bcs)
    problem = NonlinearVariationalProblem(F, solution)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h, lambda_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_lsh_alternative(
    mesh, 
    degree=1, 
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        trace_family = "HDiv Trace"
        T = FunctionSpace(mesh, trace_family, degree)
    W = U * V * T

    # Trial and test functions
    solution = Function(W)
    u, p, lambda_h = split(solution)
    # u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # BCs
    p_exact = exact_trace
    # bcs = DirichletBC(W.sub(2), exact_trace, "on_boundary")
    bcs = DirichletBC(W.sub(2), p_exact, "on_boundary")

    # Hybridization parameter
    beta_0 = Constant(1.0e0)
    beta = beta_0 / h
    # beta = beta_0
    beta_avg = beta_0 / h("+")

    # Stabilizing parameter
    # delta_0 = Constant(1)
    # delta_1 = Constant(1)
    # delta_2 = Constant(1)
    # delta_3 = Constant(1)
    # delta_4 = Constant(1)
    # delta_5 = Constant(1)
    # delta = h * h
    delta = Constant(1.0)
    # delta = 1 / h
    # delta_0 = delta
    delta_0 = Constant(1)
    # delta_1 = delta
    delta_1 = Constant(0)  #* delta
    delta_2 = delta
    delta_3 = delta
    beta_1 = Constant(0.0e0)
    delta_4 = beta_1 / h
    delta_5 = delta * Constant(0)  #/ h
    # delta_5 = LARGE_NUMBER / h

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n
    v_hat = v + beta * (q - mu_h) * n

    # Flux least-squares
    # Comments: For this expanded form, when using every parameter as one (note that edge terms in flux contribution should match
    # with transmission condition weighting), super convergence was achieved for triangles for degree = 1.
    # It is important to note that, in such a case, beta is not zero, so u_hat edge stabilization should be included.
    a = (
        (inner(u, v) - q * div(u) - p * div(v) + inner(grad(p), grad(q)))
        * delta_0
        * dx
    )
    # These terms below are unsymmetric
    a += delta_0("+") * jump(u_hat, n=n) * q("+") * dS
    a += delta_0 * dot(u, n) * q * ds
    a += delta_0 * beta * (p - lambda_h) * q * ds
    a += delta_0("+") * lambda_h("+") * jump(v, n=n) * dS
    a += delta_0 * lambda_h * dot(v, n) * ds
    # L = delta_0 * exact_solution * dot(v, n) * ds

    # Flux Least-squares as in DG
    # a = delta_0 * inner(u + grad(p), v + grad(q)) * dx

    # Classical mixed Darcy eq. first-order terms as stabilizing terms
    a += delta_1 * (inner(u, v) - div(v) * p) * dx
    a += delta_1("+") * lambda_h("+") * jump(v, n=n) * dS
    a += delta_1 * lambda_h * dot(v, n) * ds

    # Mass balance least-square
    a += delta_2 * div(u) * div(v) * dx
    L = delta_2 * f * div(v) * dx

    # Irrotational least-squares
    a += delta_3 * inner(curl(u), curl(v)) * dx

    # Hybridization terms
    a += mu_h("+") * jump(u_hat, n=n) * dS
    a += mu_h * (lambda_h - p_exact) * ds
    # a += mu_h * dot(u_hat, n) * ds
    # L += mu_h * dot(sigma_e, n) * ds
    a += delta_4("+") * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
    a += delta_4 * (p - p_exact) * q * ds
    a += delta_4 * (lambda_h - p_exact) * mu_h * ds
    # a += delta_4 * (exact_solution - lambda_h) * (q - mu_h) * ds
    # Alternative primal
    # a += delta_4("+") * (lambda_h("+") - p("+")) * (mu_h("+") - q("+")) * dS
    # a += delta_4 * (lambda_h - p) * (mu_h - q) * ds
    # Flux
    a += delta_5("+") * (dot(u, n)("+") - dot(u_hat, n)("+")) * (dot(v, n)("+") - dot(v_hat, n)("+")) * dS
    a += delta_5 * (dot(u, n) - dot(u_hat, n)) * (dot(v, n) - dot(v_hat, n)) * ds
    # Alternative
    # a += delta_5("+") * (dot(u_hat, n)("+") - dot(u, n)("+")) * (dot(v_hat, n)("+") - dot(v, n)("+")) * dS
    # a += delta_5 * (dot(u_hat, n) - dot(u, n)) * (dot(v_hat, n) - dot(v, n)) * ds

    # Weakly imposed BC from hybridization
    # a += mu_h * (lambda_h - exact_trace) * ds
    # a += mu_h * lambda_h * ds
    # ###
    # a += (
    #     delta_4 * (mu_h - q) * (lambda_h - exact_solution) * ds
    # )  # maybe this is not a good way to impose BC, but this necessary
    # a += (
    #     delta_4 * (q - mu_h) * (exact_solution - lambda_h) * ds
    # )  # maybe this is not a good way to impose BC, but this necessary
    L += delta_1 * exact_solution * dot(v, n) * ds  # study if this is a good BC imposition

    F = a - L

    params = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "pmat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        # Use the static condensation PC for hybridized problems
        # and use a direct solve on the reduced system for lambda_h
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0, 1",
        "condensed_field": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }
    # problem = NonlinearVariationalProblem(F, solution, bcs=bcs)
    problem = NonlinearVariationalProblem(F, solution)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h, lambda_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_lsh_expanded(
    mesh, 
    degree=1, 
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        trace_family = "HDiv Trace"
        T = FunctionSpace(mesh, trace_family, degree)
    W = U * V * T

    # Trial and test functions
    solution = Function(W)
    u, p, lambda_h = split(solution)
    # u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # BCs
    p_exact = exact_trace

    # Hybridization parameter
    beta_0 = Constant(1.0e0)
    beta = beta_0 / h
    # beta = beta_0

    # Stabilizing parameter
    delta = Constant(1.0)
    delta_0 = Constant(1)
    delta_1 = Constant(0)  #* delta
    delta_2 = delta
    delta_3 = delta
    beta_1 = Constant(0.0e0)
    delta_4 = beta_1 / h
    # delta_5 = LARGE_NUMBER / h

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # Flux least-squares
    # Comments: For this expanded form, when using every parameter as one (note that edge terms in flux contribution should match
    # with transmission condition weighting), super convergence was achieved for triangles for degree = 1.
    # It is important to note that, in such a case, beta is not zero, so u_hat edge stabilization should be included.
    a = (
        (inner(u, v) - q * div(u) - p * div(v) + inner(grad(p), grad(q)))
        * delta_0
        * dx
    )
    # These terms below are unsymmetric
    a += delta_0("+") * jump(u_hat, n=n) * q("+") * dS
    a += delta_0 * dot(u, n) * q * ds
    a += delta_0 * beta * (p - p_exact) * q * ds
    a += delta_0("+") * lambda_h("+") * jump(v, n=n) * dS
    # a += delta_0 * lambda_h * dot(v, n) * ds
    L = -delta_0 * p_exact * dot(v, n) * ds

    # Flux Least-squares as in DG
    # a += delta_0 * inner(u + grad(p), v + grad(q)) * dx

    # Classical mixed Darcy eq. first-order terms as stabilizing terms
    a += delta_1 * (inner(u, v) - div(v) * p) * dx
    a += delta_1("+") * lambda_h("+") * jump(v, n=n) * dS
    a += delta_1 * lambda_h * dot(v, n) * ds
    L += delta_1 * p_exact * dot(v, n) * ds  # study if this is a good BC imposition

    # Mass balance least-square
    a += delta_2 * div(u) * div(v) * dx
    L += delta_2 * f * div(v) * dx

    # Irrotational least-squares
    a += delta_3 * inner(curl(u), curl(v)) * dx

    # Hybridization terms
    a += mu_h("+") * jump(u_hat, n=n) * dS
    a += mu_h * (lambda_h - p_exact) * ds
    a += Constant(0) * delta_4("+") * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
    a += delta_4 * (p - p_exact) * q * ds
    a += delta_4 * (lambda_h - p_exact) * mu_h * ds

    F = a - L

    params = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "pmat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        # Use the static condensation PC for hybridized problems
        # and use a direct solve on the reduced system for lambda_h
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0, 1",
        "condensed_field": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }
    # problem = NonlinearVariationalProblem(F, solution, bcs=bcs)
    problem = NonlinearVariationalProblem(F, solution)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h, lambda_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_lsh_embedded_mass(
    mesh, 
    degree=1, 
    is_multiplier_continuous=False
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        trace_family = "HDiv Trace"
        T = FunctionSpace(mesh, trace_family, degree)
    W = U * V * T

    # Trial and test functions
    solution = Function(W)
    u, p, lambda_h = split(solution)
    # u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # BCs
    p_exact = exact_trace

    # Hybridization parameter
    beta_0 = Constant(0.0e0)
    beta = beta_0 / h
    # beta = beta_0

    # Stabilizing parameter
    delta = Constant(0.5)
    delta_0 = delta * Constant(1) * h * h
    delta_1 = Constant(1)  #* delta
    delta_2 = delta * h * h
    delta_3 = delta * h * h
    beta_1 = Constant(1.0e0)
    delta_4 = beta_1 / h
    delta_5 = delta * Constant(0)  #/ h
    # delta_5 = LARGE_NUMBER / h

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # Flux least-squares
    # Comments: For this expanded form, when using every parameter as one (note that edge terms in flux contribution should match
    # with transmission condition weighting), super convergence was achieved for triangles for degree = 1.
    # It is important to note that, in such a case, beta is not zero, so u_hat edge stabilization should be included.
    a = (
        (inner(u, v) - q * div(u) - p * div(v) + inner(grad(p), grad(q)))
        * delta_0
        * dx
    )
    # These terms below are unsymmetric
    a += delta_0("+") * jump(u_hat, n=n) * q("+") * dS
    a += delta_0 * dot(u, n) * q * ds
    a += delta_0 * beta * (p - p_exact) * q * ds
    a += delta_0("+") * lambda_h("+") * jump(v, n=n) * dS
    a += delta_0 * lambda_h * dot(v, n) * ds
    # L = delta_0 * exact_solution * dot(v, n) * ds

    # Flux Least-squares as in DG
    # a = delta_0 * inner(u + grad(p), v + grad(q)) * dx

    # Classical mass residual term
    a += -delta_1 * dot(u, grad(q)) * dx
    a += delta_1("+") * q("+") * jump(u_hat, n=n) * dS
    a += delta_1 * dot(u, n) * q * ds
    a += delta_1 * beta * (p - lambda_h) * q * ds
    L = delta_1 * f * q * dx

    # Mass balance least-square
    a += delta_2 * div(u) * div(v) * dx
    L += delta_2 * f * div(v) * dx

    # Irrotational least-squares
    a += delta_3 * inner(curl(u), curl(v)) * dx

    # Hybridization terms
    a += mu_h("+") * jump(u_hat, n=n) * dS
    a += mu_h * (lambda_h - p_exact) * ds
    a += delta_4("+") * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
    a += delta_4 * (p - p_exact) * q * ds
    a += delta_4 * (lambda_h - p_exact) * mu_h * ds

    F = a - L

    params = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "pmat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        # Use the static condensation PC for hybridized problems
        # and use a direct solve on the reduced system for lambda_h
        "pc_python_type": "firedrake.SCPC",
        "pc_sc_eliminate_fields": "0, 1",
        "condensed_field": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
    }
    # problem = NonlinearVariationalProblem(F, solution, bcs=bcs)
    problem = NonlinearVariationalProblem(F, solution)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    sigma_h, p_h, lambda_h = solution.split()
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def solve_poisson_lsh_primal(
    mesh, 
    degree=1, 
    is_multiplier_continuous=True
):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    pressure_family = 'DQ' if use_quads else 'DG'
    velocity_family = 'DQ' if use_quads else 'DG'
    U = VectorFunctionSpace(mesh, velocity_family, degree)
    p_degree = degree
    V = FunctionSpace(mesh, pressure_family, p_degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        trace_family = "HDiv Trace"
        T = FunctionSpace(mesh, trace_family, degree)
    W = V * T

    # Trial and test functions
    solution = Function(W)
    p, lambda_h = split(solution)
    # u, p, lambda_h = TrialFunctions(W)
    q, mu_h  = TestFunctions(W)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e, exact_trace = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Forcing function
    f = div(-grad(exact_solution))

    # BCs
    p_exact = exact_trace
    # bcs = DirichletBC(W.sub(2), exact_trace, "on_boundary")
    bcs = DirichletBC(W.sub(1), p_exact, "on_boundary")

    # Hybridization parameter
    penalty_constant = 1e1
    beta_0 = Constant(penalty_constant * p_degree * p_degree) * Constant(1)
    beta = beta_0 / h

    # Stabilizing parameter (working perfectly)
    delta_base = h * h
    delta_0 = Constant(1)
    delta_1 = delta_base * Constant(1)
    delta_2 = Constant(penalty_constant * p_degree * p_degree) / h
    
    # Stabilizing parameter (tests)
    delta_base = h * h
    delta_0 = Constant(1)
    delta_1 = delta_base * Constant(1)
    # delta_2 = delta_base * Constant(6 * degree) / h
    delta_2 = Constant(penalty_constant * p_degree * p_degree) / delta_base
    # delta_2 = delta_1
    # delta_2 = beta
    # delta_2 = delta_base * Constant(1) / h

    # Flux variables
    u = -grad(p)
    v = -grad(q)

    # Symmetry parameter: s = 1 (symmetric) or s = -1 (unsymmetric). Disable with 0.
    s = Constant(1)

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # Classical term
    a = delta_0 * dot(grad(p), grad(q)) * dx + delta_0('+') * jump(u_hat, n) * q("+") * dS
    # a += delta_0 * dot(u_hat, n) * q * ds
    a += delta_0 * dot(u, n) * q * ds +  delta_0 * beta * (p - exact_solution) * q * ds  # expand u_hat product in ds
    L = delta_0 * f * q * dx

    # Mass balance least-squares
    a += delta_1 * div(u) * div(v) * dx
    a += delta_1 * inner(curl(u), curl(v)) * dx
    L += delta_1 * f * div(v) * dx

    # Hybridization terms
    a += -mu_h("+") * jump(u_hat, n=n) * dS
    # a += mu_h * (lambda_h - p_exact) * ds
    # a += mu_h * dot(u_hat - grad(exact_solution), n) * ds  # is this worthy?
    a += delta_2("+") * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
    # a += delta_2 * (p - p_exact) * (q - mu_h) * ds
    a += delta_2 * (p - exact_solution) * q * ds  # needed if not included as strong BC
    a += delta_2 * (lambda_h - p_exact) * mu_h * ds  # needed if not included as strong BC
    # a += (
    #     delta_3 * (q - mu_h) * (exact_solution - lambda_h) * ds
    # )  # maybe this is not a good way to impose BC, but this necessary

    # Consistent symmetrization
    a += s * delta_0 * jump(v, n) * (p('+') - lambda_h("+")) * dS
    a += s * delta_0 * dot(v, n) * (p - exact_solution) * ds

    F = a - L

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
            "pc_factor_mat_solver_type": "mumps",
            # "mat_mumps_icntl_4": "2",
        },
    }
    # problem = NonlinearVariationalProblem(F, solution, bcs=bcs)
    problem = NonlinearVariationalProblem(F, solution)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    solver.solve()

    # Retrieving the solution
    p_h, lambda_h = solution.split()
    sigma_h = Function(U, name='Velocity')
    sigma_h.project(-grad(p_h))
    sigma_h.rename('Velocity', 'label')
    p_h.rename('Pressure', 'label')

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")

    return p_error_L2, p_error_H1, sigma_error_L2, sigma_error_Hdiv


def compute_convergence_hp(
    solver,
    min_degree=1,
    max_degree=4,
    numel_xy=(2, 4, 8, 16, 32, 64, 128, 256),
    quadrilateral=True,
    name="",
    **kwargs
):
    computed_errors_dict = {
        "Element": list(),
        "Degree": list(),
        "Cells": list(),
        "log Cells": list(),
        "Mesh size": list(),
        "L2-error p": list(),
        "log L2-error p": list(),
        "L2-error p order": list(),
        "H1-error p": list(),
        "log H1-error p": list(),
        "H1-error p order": list(),
        "L2-error u": list(),
        "log L2-error u": list(),
        "L2-error u order": list(),
        "Hdiv-error u": list(),
        "Hdiv-error u order": list(),
    }
    element_kind = "Quad" if quadrilateral else "Tri"
    for degree in range(min_degree, max_degree):
        p_errors_L2 = np.array([])
        p_errors_H1 = np.array([])
        v_errors_L2 = np.array([])
        v_errors_Hdiv = np.array([])
        num_cells = np.array([])
        mesh_size = np.array([])
        for n in numel_xy:
            nel_x = nel_y = n
            mesh = UnitSquareMesh(nel_x, nel_y, quadrilateral=quadrilateral)
            current_num_cells = mesh.num_cells()
            num_cells = np.append(num_cells, current_num_cells)
            current_mesh_size = mesh.cell_sizes.dat.data_ro.min() if not quadrilateral else 1 / n
            mesh_size = np.append(mesh_size, current_mesh_size)

            (
                current_error_p_L2, 
                current_error_p_H1, 
                current_error_v_L2, 
                current_error_v_Hdiv) = solver(mesh=mesh, degree=degree, **kwargs)

            p_errors_L2 = np.append(p_errors_L2, current_error_p_L2)
            p_errors_H1 = np.append(p_errors_H1, current_error_p_H1)
            v_errors_L2 = np.append(v_errors_L2, current_error_v_L2)
            v_errors_Hdiv = np.append(v_errors_Hdiv, current_error_v_Hdiv)

            computed_errors_dict["Element"].append(element_kind)
            computed_errors_dict["Degree"].append(degree)
            computed_errors_dict["Cells"].append(current_num_cells)
            computed_errors_dict["log Cells"].append(np.log10(current_num_cells) / 2)
            computed_errors_dict["Mesh size"].append(current_mesh_size)
            computed_errors_dict["L2-error p"].append(current_error_p_L2)
            computed_errors_dict["log L2-error p"].append(np.log10(current_error_p_L2))
            computed_errors_dict["H1-error p"].append(current_error_p_H1)
            computed_errors_dict["log H1-error p"].append(np.log10(current_error_p_H1))
            computed_errors_dict["L2-error u"].append(current_error_v_L2)
            computed_errors_dict["log L2-error u"].append(np.log10(current_error_v_L2))
            computed_errors_dict["Hdiv-error u"].append(current_error_v_Hdiv)

        p_L2_errors_log2 = np.log10(p_errors_L2)
        p_H1_errors_log2 = np.log10(p_errors_H1)
        v_L2_errors_log2 = np.log10(v_errors_L2)
        v_Hdiv_errors_log2 = np.log10(v_errors_Hdiv)
        num_cells_log2 = np.log10(num_cells)
        mesh_size_log2 = np.log10(mesh_size)

        PETSc.Sys.Print("\n--------------------------------------")

        p_slope_L2, _, _, _, _ = linregress(mesh_size_log2, p_L2_errors_log2)
        PETSc.Sys.Print(
            "\nDegree %d: p slope L2-error %f"
            % (degree, np.abs(p_slope_L2))
        )

        p_slope_H1, _, _, _, _ = linregress(mesh_size_log2, p_H1_errors_log2)
        PETSc.Sys.Print(
            "\nDegree %d: p slope H1-error %f"
            % (degree, np.abs(p_slope_H1))
        )

        v_slope_L2, _, _, _, _ = linregress(
            mesh_size_log2, v_L2_errors_log2
        )
        PETSc.Sys.Print(
            "\nDegree %d: v slope L2-error %f"
            % (degree, np.abs(v_slope_L2))
        )

        v_slope_Hdiv, _, _, _, _ = linregress(
            mesh_size_log2, v_Hdiv_errors_log2
        )
        PETSc.Sys.Print(
            "\nDegree %d: v slope Hdiv-error %f"
            % (degree, np.abs(v_slope_Hdiv))
        )

        num_mesh_evaluations = len(numel_xy)
        computed_errors_dict["L2-error p order"] += num_mesh_evaluations * [np.abs(p_slope_L2)]
        computed_errors_dict["H1-error p order"] += num_mesh_evaluations * [np.abs(p_slope_H1)]
        computed_errors_dict["L2-error u order"] += num_mesh_evaluations * [np.abs(v_slope_L2)]
        computed_errors_dict["Hdiv-error u order"] += num_mesh_evaluations * [np.abs(v_slope_Hdiv)]

        PETSc.Sys.Print("\n--------------------------------------")

    dir_name = f"./conv_rate_results/conv_results_{name}"
    os.makedirs(dir_name, exist_ok=True)
    df_computed_errors = pd.DataFrame(data=computed_errors_dict)
    path_to_save_results = f"{dir_name}/errors.csv"
    df_computed_errors.to_csv(path_to_save_results)

    return


# Solver options
available_solvers = {
    # "cg": solve_poisson_cg,
    # "cgls": solve_poisson_cgls,  # Compare
    # "dgls": solve_poisson_dgls,
    # "sdhm": solve_poisson_sdhm,  # Compare
    # "ls": solve_poisson_ls,  # Compare
    # "dls": solve_poisson_dls,
    # "lsh": solve_poisson_lsh,  # Compare
    # "lsh_alternative": solve_poisson_lsh_alternative,  # Compare
    # "lsh_expanded": solve_poisson_lsh_expanded,  # Compare
    # "lsh_embedded_mass": solve_poisson_lsh_embedded_mass,  # Compare
    "new_lsh_primal": solve_poisson_lsh_primal,
    # "dls_ip_primal": solve_poisson_dls_primal,
    # "cls_ip_primal": solve_poisson_cls_primal,
    # "vms": solve_poisson_vms,  # Compare
    # "dvms": solve_poisson_dvms,  # Compare
    # "mixed_RT": solve_poisson_mixed_RT,  # Compare
    # "hdg": solve_poisson_hdg,  # Compare
    # "cgh": solve_poisson_cgh,
    # "ldgc": solve_poisson_ldgc,
    # "sipg": solve_poisson_sipg,
}

degree = 1
last_degree = 4
# mesh_quad = [False, True]  # Triangles, Quads
mesh_quad = [False]
# elements_for_each_direction = [15, 20, 25, 30, 35, 40]
# elements_for_each_direction = [4, 8, 16, 32, 64, 128]  # for PhD
elements_for_each_direction = [4, 8, 16, 32, 64]
# elements_for_each_direction = [4, 6, 8, 10, 12, 14]
for element in mesh_quad:
    for current_solver in available_solvers:

        if element:
            element_kind = "quad"
        else:
            element_kind = "tri"

        # Setting the output file name
        name = f"{current_solver}_{element_kind}"

        # Selecting the solver
        solver = available_solvers[current_solver]

        PETSc.Sys.Print("*******************************************\n")
        PETSc.Sys.Print(f"*** Begin case: {name} ***\n")

        # Performing the convergence study
        compute_convergence_hp(
            solver,
            min_degree=degree,
            max_degree=last_degree + 1,
            quadrilateral=element,
            numel_xy=elements_for_each_direction,
            name=name
        )

        PETSc.Sys.Print(f"\n*** End case: {name} ***")
        PETSc.Sys.Print("*******************************************\n")
