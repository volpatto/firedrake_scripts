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
from typing import Dict, Any
import os

matplotlib.use('Agg')


@attr.s
class MatrixInfo(object):
    petsc_mat: Any
    is_symmetric: bool
    size: int
    nnz: int
    number_of_dofs: int


@attr.s
class ApproximationErrorResult(object):
    errors_values: Dict[str, float]
    degree: int
    number_of_dofs: int
    cell_size: float


def calculate_max_mesh_size(mesh: Mesh) -> float:
    """
    Convenient function to compute the max of the mesh size.
    
    This is collected from a Firedrake discussion.
    See here: https://github.com/firedrakeproject/firedrake/discussions/2547
    """
    P0 = FunctionSpace(mesh, "DG", 0)
    hmax = project(CellSize(mesh), P0)
    with hmax.dat.vec_ro as v:
        global_hmax = v.max()
        
    return global_hmax


def norm_trace(v, norm_type="L2", mesh=None):
    r"""Compute the norm of ``v``.

    :arg v: a ufl expression (:class:`~.ufl.classes.Expr`) to compute the norm of
    :arg norm_type: the type of norm to compute, see below for
         options.
    :arg mesh: an optional mesh on which to compute the norm
         (currently ignored).

    Available norm types are:

    - Lp :math:`||v||_{L^p} = (\int |v|^p)^{\frac{1}{p}} \mathrm{d}s`

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
    else:
        raise RuntimeError("Unknown norm type '%s'" % norm_type)

    return assemble((expr("+")**(p/2))*dS)**(1/p) + assemble((expr**(p/2))*ds)**(1/p)


def errornorm_trace(u, uh, norm_type="L2", degree_rise=None, mesh=None):
    """Compute the error :math:`e = u - u_h` in the specified norm on the mesh skeleton.

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


def exact_solutions_expressions(mesh):
    x, y = SpatialCoordinate(mesh)
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)  # original
    # p_exact = sin(0.5 * pi * x) * sin(0.5 * pi * y)
    # p_exact = 0.5 / (pi * pi) * cos(pi * x) * cos(pi * y)  # Nunez
    # p_exact = x * x * x - 3 * x * y * y
    # p_exact = - (x * x / 2 - x * x * x / 3) * (y * y / 2 - y * y * y / 3)
    flux_exact = -grad(p_exact)
    return p_exact, flux_exact


def calculate_exact_solution(mesh, pressure_family, velocity_family, pressure_degree, velocity_degree, is_hdiv_space=False):
    '''
    For compatibility only. Should be removed.
    '''
    return exact_solutions_expressions(mesh)


def calculate_exact_solution_with_trace(mesh, pressure_family, velocity_family, pressure_degree, velocity_degree):
    '''
    For compatibility only. Should be removed.
    '''
    return exact_solutions_expressions(mesh)


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
    element_family = 'DQ' if use_quads else 'DG'
    # DiscontinuousElement = FiniteElement(element_family, mesh.ufl_cell(), degree)
    S = FiniteElement("S", mesh.ufl_cell(), degree)
    DiscontinuousElement = BrokenElement(S)
    U = VectorFunctionSpace(mesh, DiscontinuousElement)
    V = FunctionSpace(mesh, DiscontinuousElement)

    # Trial and test functions
    p = TrialFunction(V)
    q = TestFunction(V)

    # Mesh entities
    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    exact_solution, sigma_e = exact_solutions_expressions(mesh)

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
    enable_dg_ip = Constant(1)  # enable (1) or disable (0)
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
    # penalty_constant = 1e1
    # penalty_constant_ip = 0e2
    # # delta_base = Constant(penalty_constant * degree * degree)
    # delta_base = Constant(1)
    # # delta_base = Constant(penalty_constant)
    # enable_dg_ip = Constant(0)  # enable (1) or disable (0)
    # delta_0 = delta_base / delta_base * enable_dg_ip
    # delta_1 = Constant(1)
    # # delta_1 = Constant(1)
    # delta_2 = delta_base / h / h * Constant(1)
    # # delta_2 = delta_base / h
    # # delta_2 = Constant(1)
    # delta_3 = Constant(1) / delta_base
    
    # penalty_constant = 1e4 * degree * degree
    # penalty_constant = 1e6 * degree * degree  # good
    penalty_constant = 1e1
    penalty_constant_ip = penalty_constant
    delta_base = Constant(penalty_constant)
    enable_dg_ip = Constant(0)  # enable (1) or disable (0)
    delta_0 = delta_base / delta_base * enable_dg_ip
    # delta_1 = Constant(1) * delta_base * h * h
    delta_1 = Constant(1) * h
    # delta_2 = delta_base / h / h
    delta_2 = delta_base / h * Constant(1)
    # delta_3 = Constant(1)
    delta_3 = (1 / delta_base) * h * Constant(1)
    delta_4 = delta_2
    # delta_4 = Constant(1e8) / h
    # delta_3 = delta_base * h

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
    a += delta_4 * (p - exact_solution) * q * ds
    a += avg(delta_3) * jump(u, n=n) * jump(v, n=n) * dS
    # Flux should not be imposed at Dirichlet condition boundaries
    # a += delta_3 * dot(u, n) * dot(v, n) * ds
    # L += delta_3 * dot(sigma_e, n) * dot(v, n) * ds

    # DG-IP Weak boundary conditions (not required, already imposed by LS terms)
    # beta0 = Constant(enable_dg_ip * penalty_constant_ip)
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
        "mat_mumps_icntl_14": "1000",
        # "pc_factor_mat_solver_type": "petsc",
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
    
    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
    }
    num_dofs = V.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        errors_values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_interpolator_primal(mesh, degree=1):
    # Function space declaration
    use_quads = str(mesh.ufl_cell()) == "quadrilateral"
    element_family = 'DQ' if use_quads else 'DG'
    DiscontinuousElement = FiniteElement(element_family, mesh.ufl_cell(), degree)
    # CG = FiniteElement("CG", mesh.ufl_cell(), degree)
    # CG = CG['facet']
    # DiscontinuousElement = BrokenElement(CG)
    # S = FiniteElement("S", mesh.ufl_cell(), degree)
    # DiscontinuousElement = BrokenElement(S)
    U = VectorFunctionSpace(mesh, DiscontinuousElement)
    V = FunctionSpace(mesh, DiscontinuousElement)

    # Exact solution
    exact_solution, sigma_e = exact_solutions_expressions(mesh)

    # Interpolating the solution
    p_h = interpolate(exact_solution, V)
    sigma_h = interpolate(-grad(p_h), U)

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

    # Calculating H1-error for primal variable
    p_error_H1 = errornorm(exact_solution, p_h, norm_type="H1")

    # Calculating L2-error for flux variable
    sigma_error_L2 = errornorm(sigma_e, sigma_h, norm_type="L2")

    # Calculating Hdiv-error for flux variable
    sigma_error_Hdiv = errornorm(sigma_e, sigma_h, norm_type="Hdiv")
    
    errors = {
        'L2-error p': p_error_L2,
        'H1-error p': p_error_H1,
        'L2-error u': sigma_error_L2,
        'Hdiv-error u': sigma_error_Hdiv,
    }
    num_dofs = V.dim()
    max_mesh_size = calculate_max_mesh_size(mesh)
    
    approximation_error_results = ApproximationErrorResult(
        errors_values=errors, degree=degree, number_of_dofs=num_dofs, cell_size=max_mesh_size
    )

    return approximation_error_results


def solve_poisson_interpolator_mixed(mesh, degree=1, is_multiplier_continuous=False):
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

    # Exact solution
    exact_solution, sigma_e, p_expression = calculate_exact_solution_with_trace(
        mesh, 
        pressure_family, 
        velocity_family, 
        degree + 3, 
        degree + 3
    )

    # Interpolating the solution
    p_h = interpolate(exact_solution, V)
    lambda_h = interpolate(exact_solution, T)
    sigma_h = interpolate(sigma_e, U)

    # Calculating L2-error for primal variable
    p_error_L2 = errornorm_trace(project(p_expression, T), lambda_h, norm_type="L2")
    # p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")

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

    # Hybridization parameter
    beta_0 = Constant(0.0e0)
    beta = beta_0 / h
    # beta = beta_0

    # Stabilization parameters
    delta_0 = Constant(-1)
    delta_1 = Constant(-0.5)  #* h * h
    delta_2 = Constant(0.5)
    delta_3 = Constant(0.5)

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
    a += dot(u, n) * mu_h * ds
    a += -lambda_h * mu_h * ds  # Classical required term
    L += -exact_solution * mu_h * ds  # Pair for the above classical required term

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
    # p_error_L2 = errornorm(exact_solution, p_h, norm_type="L2")
    # interpolate(exact_trace, T)
    p_error_L2 = errornorm_trace(project(exact_trace, T), lambda_h, norm_type="L2")

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


def solve_poisson_hdg_sdhm(
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

    # Hybridization parameter
    beta_0 = Constant(1.0e1)
    beta = beta_0 / h
    # beta = beta_0

    # Stabilization parameters
    delta_1 = Constant(-0.5)  #* h * h
    delta_2 = Constant(0.5) * h * h
    delta_3 = Constant(0.5) * h * h

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # HDG classical form
    a = (dot(u, v) - div(v) * p) * dx + lambda_h("+") * jump(v, n) * dS
    a += -dot(u, grad(q)) * dx + jump(u_hat, n) * q("+") * dS
    L = f * q * dx

    # Least-squares terms
    a += delta_1 * inner(u + grad(p), v + grad(q)) * dx
    a += delta_2 * div(u) * div(v) * dx
    a += delta_3 * inner(curl(u), curl(v)) * dx
    L += delta_2 * f * div(v) * dx

    # Transmission condition
    a += jump(u_hat, n) * mu_h("+") * dS

    # Weakly imposed BC
    # a += lambda_h * dot(v, n) * ds  # required term
    L += -exact_solution * dot(v, n) * ds  # required as the above, but just one of them should be used (works for continuous multiplier)
    a += dot(u, n) * q * ds
    a += dot(u, n) * mu_h * ds
    a += beta * p * q * ds  # required term... note that u (the unknown) is used
    L += beta * exact_solution * q * ds  # Required, this one is paired with the above term
    a += -lambda_h * mu_h * ds  # Classical required term
    L += -exact_solution * mu_h * ds  # Pair for the above classical required term

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
    reorder_mesh=False,
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
            mesh = UnitSquareMesh(nel_x, nel_y, quadrilateral=quadrilateral, reorder=reorder_mesh)
            current_num_cells = mesh.num_cells()
            num_cells = np.append(num_cells, current_num_cells)
            current_mesh_size = mesh.cell_sizes.dat.data_ro.min() if not quadrilateral else 1 / n
            mesh_size = np.append(mesh_size, current_mesh_size)

            current_errors = solver(mesh=mesh, degree=degree, **kwargs)

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

        p_L2_errors_log10 = np.log10(p_errors_L2)
        p_H1_errors_log10 = np.log10(p_errors_H1)
        v_L2_errors_log10 = np.log10(v_errors_L2)
        v_Hdiv_errors_log10 = np.log10(v_errors_Hdiv)
        num_cells_log10 = np.log10(num_cells)
        mesh_size_log10 = np.log10(mesh_size)

        PETSc.Sys.Print("\n--------------------------------------")

        p_slope_L2, _, _, _, _ = linregress(mesh_size_log10, p_L2_errors_log10)
        PETSc.Sys.Print(
            "\nDegree %d: p slope L2-error %f"
            % (degree, np.abs(p_slope_L2))
        )

        p_slope_H1, _, _, _, _ = linregress(mesh_size_log10, p_H1_errors_log10)
        PETSc.Sys.Print(
            "\nDegree %d: p slope H1-error %f"
            % (degree, np.abs(p_slope_H1))
        )

        v_slope_L2, _, _, _, _ = linregress(
            mesh_size_log10, v_L2_errors_log10
        )
        PETSc.Sys.Print(
            "\nDegree %d: v slope L2-error %f"
            % (degree, np.abs(v_slope_L2))
        )

        v_slope_Hdiv, _, _, _, _ = linregress(
            mesh_size_log10, v_Hdiv_errors_log10
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
    # "hdg_sdhm": solve_poisson_hdg_sdhm,
    # "ls": solve_poisson_ls,  # Compare
    # "dls": solve_poisson_dls,
    # "lsh": solve_poisson_lsh,  # Compare
    # "lsh_alternative": solve_poisson_lsh_alternative,  # Compare
    # "lsh_expanded": solve_poisson_lsh_expanded,  # Compare
    # "lsh_embedded_mass": solve_poisson_lsh_embedded_mass,  # Compare
    # "new_lsh_primal": solve_poisson_lsh_primal,
    "dls_ip_primal": solve_poisson_dls_primal,
    # "interpolator_primal": solve_poisson_interpolator_primal,  # Compare
    # "interpolator_mixed": solve_poisson_interpolator_mixed,  # Compare
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
mesh_quad = [True]
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
            name=name,
            reorder_mesh=True
        )

        PETSc.Sys.Print(f"\n*** End case: {name} ***")
        PETSc.Sys.Print("*******************************************\n")
