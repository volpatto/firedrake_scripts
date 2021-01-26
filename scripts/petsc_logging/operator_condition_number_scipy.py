import attr
from firedrake import *
import numpy as np
from logging import warning
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse.linalg import svds, eigs, ArpackNoConvergence
from scipy.sparse import csr_matrix
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
    A, num_of_factors, 
    backend: str = "scipy", 
    imag_threshold: float = 1e-5,
    zero_tol: float = 1e-10,
    is_negative_spectrum: bool = False
):
    backend = backend.lower()

    if backend == "scipy":
        size = A.getSize()
        Mnp = csr_matrix(A.getValuesCSR()[::-1], shape=size)
        Mnp.eliminate_zeros()

        largest_eigenvalues = eigs(Mnp, k=num_of_factors, which="LM", return_eigenvectors=False)
        real_largest_eigenvalues = filter_real_part_in_array(largest_eigenvalues, imag_threshold)

        smallest_eigenvalues = eigs(Mnp, k=num_of_factors, which="SM", return_eigenvectors=False)
        real_smallest_eigenvalues = filter_real_part_in_array(smallest_eigenvalues, imag_threshold)

        smallest_eigenvalues = real_smallest_eigenvalues[real_smallest_eigenvalues > zero_tol]
        largest_eigenvalues = real_largest_eigenvalues[real_largest_eigenvalues > zero_tol]
        condition_number = largest_eigenvalues.max() / smallest_eigenvalues.min()
    elif backend == "slepc":
        S = SLEPc.EPS()
        S.create()
        S.setOperators(A)
        is_operator_symmetric = A.isSymmetric(tol=1e-8)
        if is_operator_symmetric:
            S.setProblemType(SLEPc.EPS.ProblemType.HEP)
        else:
            S.setProblemType(SLEPc.EPS.ProblemType.NHEP)
        S.setDimensions(nev=num_of_factors, mpd=num_of_factors)
        # S.setKrylovSchurDetectZeros(detect=True)
        S.setTolerances(max_it=2000)
        S.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
        S.solve()

        nconv = S.getConverged()
        eigenvalues_list = list()
        if nconv > 0:
            for i in range(nconv):
                eigenval = S.getEigenvalue(i)
                eigenvalues_list.append(eigenval)
        else:
            raise RuntimeError("SLEPc EPS has not converged.")

        eigenvalues = np.array(eigenvalues_list)
        eigenvalues = filter_real_part_in_array(eigenvalues, imag_threshold)
        if is_negative_spectrum:
            eigenvalues = np.abs(eigenvalues[np.abs(eigenvalues) > zero_tol])
        else:
            eigenvalues = eigenvalues[eigenvalues > zero_tol]

        condition_number = eigenvalues.max() / eigenvalues.min()
    else:
        raise NotImplementedError("The required method for condition number estimation is currently unavailable.")

    return condition_number


def solve_poisson_cg(num_elements_x, num_elements_y, degree=1, use_quads=False):
    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
    V = FunctionSpace(mesh, "CG", degree)

    # Trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Dirichlet BCs
    bcs = DirichletBC(V, 0.0, "on_boundary")

    # Variational form
    a = inner(grad(u), grad(v)) * dx

    A = assemble(a, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = V.dim()

    num_of_factors = int(0.95 * number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors, backend="slepc")

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

    # Mesh entities
    x, y = SpatialCoordinate(mesh)

    # Exact solution
    p_exact = sin(2 * pi * x) * sin(2 * pi * y)
    exact_solution = Function(V).interpolate(p_exact)
    exact_solution.rename("Exact pressure", "label")
    sigma_e = Function(U, name='Exact velocity')
    sigma_e.project(-grad(p_exact))

    # Dirichlet BCs
    bcs = DirichletBC(W[0], sigma_e, "on_boundary")

    # Least-squares terms
    a = inner(u + grad(p), v + grad(q)) * dx
    a += div(u) * div(v) * dx
    a += inner(curl(u), curl(v)) * dx

    A = assemble(a, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = W.dim()

    num_of_factors = int(0.95 * number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors, backend="scipy")

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
    a = (dot(u, v) - div(v) * p - q * div(u)) * dx
    # Stabilizing terms
    a += -0.5 * inner((u + grad(p)), v + grad(q)) * dx
    # a += 0.5 * h * h * div(u) * div(v) * dx
    # a += 0.5 * h * h * inner(curl(u), curl(v)) * dx
    # L += 0.5 * h * h * f * div(v) * dx
    a += 0.5 * div(u) * div(v) * dx
    a += 0.5 * inner(curl(u), curl(v)) * dx

    A = assemble(a, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = W.dim()

    num_of_factors = int(0.95 * number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors, backend="slepc")

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

    num_of_factors = int(0.95 * number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors, backend="slepc")

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

    # Mesh entities
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

    A = assemble(a, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = W.dim()

    num_of_factors = int(0.95 * number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors, backend="slepc")

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


def solve_poisson_dgls(num_elements_x, num_elements_y, degree=1, use_quads=False):
    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
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
    bcs = DirichletBC(W[0], sigma_e, "on_boundary", method="geometric")

    # Average cell size and mesh dependent stabilization
    h_avg = (h("+") + h("-")) / 2.0

    # Jump stabilizing parameters based on Badia-Codina stabilized dG method
    L0 = 1
    eta_p = L0 * h_avg  # method B in the Badia-Codina paper
    # eta_p = L0 * L0  # method D in the Badia-Codina paper
    eta_u = h_avg / L0  # method B in the Badia-Codina paper

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p - q * div(u)) * dx
    # DG terms
    a += jump(v, n) * avg(p) * dS - avg(q) * jump(u, n) * dS
    # Edge stabilizing terms
    # ** Badia-Codina based
    # a += (eta_p / h_avg) * (jump(u, n) * jump(v, n)) * dS
    # a += (eta_u / h_avg) * dot(jump(p, n), jump(q, n)) * dS
    # ** Mesh independent terms
    a += jump(u, n) * jump(v, n) * dS
    a += dot(jump(p, n), jump(q, n)) * dS
    # Volumetric stabilizing terms
    # a += 0.5 * h * h * div(u) * div(v) * dx
    # a += 0.5 * h * h * inner(curl(u), curl(v)) * dx
    # L += 0.5 * h * h * f * div(v) * dx
    a += -0.5 * inner((u + grad(p)), v + grad(q)) * dx
    a += 0.5 * div(u) * div(v) * dx
    a += 0.5 * inner(curl(u), curl(v)) * dx

    A = assemble(a, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = W.dim()

    num_of_factors = int(0.95 * number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors, backend="slepc")

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


def solve_poisson_dvms(num_elements_x, num_elements_y, degree=1, use_quads=False):
    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
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
    bcs = DirichletBC(W[0], sigma_e, "on_boundary", method="geometric")

    # Average cell size and mesh dependent stabilization
    h_avg = (h("+") + h("-")) / 2.0

    # Jump stabilizing parameters based on Badia-Codina stabilized dG method
    L0 = 1
    eta_p = L0 * h_avg  # method B in the Badia-Codina paper
    # eta_p = L0 * L0  # method D in the Badia-Codina paper
    eta_u = h_avg / L0  # method B in the Badia-Codina paper

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p + q * div(u)) * dx
    # DG terms
    a += jump(v, n) * avg(p) * dS - avg(q) * jump(u, n) * dS
    # Edge stabilizing terms
    # ** Badia-Codina based
    # a += (eta_p / h_avg) * (jump(u, n) * jump(v, n)) * dS
    # a += (eta_u / h_avg) * dot(jump(p, n), jump(q, n)) * dS
    # ** Mesh independent (original)
    a += jump(u, n) * jump(v, n) * dS  # not considered in the original paper
    a += dot(jump(p, n), jump(q, n)) * dS
    # Volumetric stabilizing terms
    a += 0.5 * inner(u + grad(p), grad(q) - v) * dx
    a += 0.5 * h * h * div(u) * div(v) * dx
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

    num_of_factors = int(0.95 * number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors, backend="slepc")

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


def solve_poisson_dls(num_elements_x, num_elements_y, degree=1, use_quads=False):
    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
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
    bcs = DirichletBC(W[0], sigma_e, "on_boundary", method="geometric")

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

    # Least-squares terms
    a = inner(u + grad(p), v + grad(q)) * dx
    a += div(u) * div(v) * dx
    a += inner(curl(u), curl(v)) * dx
    # Edge stabilizing terms
    # ** Badia-Codina based (better results) **
    a += (eta_p / h_avg) * (jump(u, n) * jump(v, n)) * dS
    a += (eta_u / h_avg) * dot(jump(p, n), jump(q, n)) * dS
    a += (eta_u_bc / h) * p * q * ds
    # ** Mesh independent **
    # a += jump(u, n) * jump(v, n) * dS
    # a += dot(jump(p, n), jump(q, n)) * dS
    # a += p * q * ds

    A = assemble(a, bcs=bcs, mat_type="aij")
    petsc_mat = A.M.handle
    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = W.dim()

    num_of_factors = int(0.95 * number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors, backend="slepc")

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
    num_elements_x, 
    num_elements_y, 
    degree=1, 
    use_quads=False,
    is_multiplier_continuous=True
):
    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
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
    # solution = Function(W)
    # u, p, lambda_h = split(solution)
    u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

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

    # Forcing function
    f_expression = div(-grad(p_exact))
    f = Function(V).interpolate(f_expression)

    # BCs
    u_projected = sigma_e
    p_boundaries = p_exact
    bcs = DirichletBC(W.sub(2), p_exact, "on_boundary")

    # Hybridization parameter
    beta_0 = Constant(1.0e-18)
    # beta = beta_0 / h
    beta = beta_0

    # Mixed classical terms
    a = (dot(u, v) - div(v) * p - q * div(u)) * dx
    L = -f * q * dx
    # Stabilizing terms
    a += -0.5 * inner(u + grad(p), v + grad(q)) * dx
    a += 0.5 * div(u) * div(v) * dx
    a += 0.5 * inner(curl(u), curl(v)) * dx
    L += 0.5 * f * div(v) * dx
    # Hybridization terms
    a += lambda_h("+") * dot(v, n)("+") * dS + mu_h("+") * dot(u, n)("+") * dS
    a += beta("+") * (lambda_h("+") - p("+")) * (mu_h("+") - q("+")) * dS
    # Weakly imposed BC
    a += (p_boundaries * dot(v, n) + mu_h * (dot(u, n) - dot(u_projected, n))) * ds
    a += beta * (lambda_h - p_boundaries) * mu_h * ds

    F = a - L
    a_form = lhs(F)

    _A = Tensor(a_form)
    A = _A.blocks
    S = A[2, 2] - A[2, :2] * A[:2, :2].inv * A[:2, 2]
    Smat = assemble(S, bcs=bcs)
    petsc_mat = Smat.M.handle

    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = Mnp.shape[0]

    num_of_factors = int(0.99 * number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors, backend="slepc", is_negative_spectrum=True)

    result = ConditionNumberResult(
        form=a,
        assembled_form=Smat,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric,
        bcs=bcs
    )
    return result


def solve_poisson_hdg(
    num_elements_x, 
    num_elements_y, 
    degree=1, 
    use_quads=False,
    is_multiplier_continuous=True
):
    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
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
    # solution = Function(W)
    # u, p, lambda_h = split(solution)
    u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

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

    # Forcing function
    f_expression = div(-grad(p_exact))
    f = Function(V).interpolate(f_expression)

    # Dirichlet BCs
    bc_multiplier = DirichletBC(W.sub(2), p_exact, "on_boundary")

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
    a_form = lhs(F)

    _A = Tensor(a_form)
    A = _A.blocks
    S = A[2, 2] - A[2, :2] * A[:2, :2].inv * A[:2, 2]
    Smat = assemble(S, bcs=bc_multiplier)
    petsc_mat = Smat.M.handle

    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = Mnp.shape[0]

    num_of_factors = int(0.99 * number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors, backend="slepc", is_negative_spectrum=True)

    result = ConditionNumberResult(
        form=a,
        assembled_form=Smat,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric,
        bcs=bc_multiplier
    )
    return result


def solve_poisson_cgh(
    num_elements_x, 
    num_elements_y, 
    degree=1, 
    use_quads=False,
    is_multiplier_continuous=True
):
    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
    pressure_family = 'DQ' if use_quads else 'DG'
    trace_family = "HDiv Trace"
    V = FunctionSpace(mesh, pressure_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        T = FunctionSpace(mesh, trace_family, degree)
    W = V * T

    # Trial and test functions
    # solution = Function(W)
    # u, p, lambda_h = split(solution)
    p, lambda_h = TrialFunctions(W)
    q, mu_h  = TestFunctions(W)

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

    # Dirichlet BCs
    bc_multiplier = DirichletBC(W.sub(1), p_exact, "on_boundary")

    # Hybridization parameter
    beta_0 = Constant(1.0e0)
    beta = beta_0 / h
    # beta = beta_0

    # Numerical flux trace
    u = -grad(p)
    u_hat = u + beta * (p - lambda_h) * n

    # HDG classical form
    a = -dot(u, grad(q)) * dx + jump(u_hat, n) * q("+") * dS
    L = f * q * dx
    # Transmission condition
    a += jump(u_hat, n) * mu_h("+") * dS
    # Weakly imposed BC
    a += dot(u_hat, n) * q * ds

    F = a - L
    a_form = lhs(F)

    _A = Tensor(a_form)
    A = _A.blocks
    S = A[1, 1] - A[1, :1] * A[:1, :1].inv * A[:1, 1]
    Smat = assemble(S, bcs=bc_multiplier)
    petsc_mat = Smat.M.handle

    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = Mnp.shape[0]

    num_of_factors = int(0.99 * number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors, backend="slepc", is_negative_spectrum=True)

    result = ConditionNumberResult(
        form=a,
        assembled_form=Smat,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric,
        bcs=bc_multiplier
    )
    return result


def solve_poisson_ldgc(
    num_elements_x, 
    num_elements_y, 
    degree=1, 
    use_quads=False,
    is_multiplier_continuous=True
):
    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
    primal_family = "DQ" if use_quads else "DG"
    V = FunctionSpace(mesh, primal_family, degree)
    if is_multiplier_continuous:
        LagrangeElement = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
        C0TraceElement = LagrangeElement["facet"]
        T = FunctionSpace(mesh, C0TraceElement)
    else:
        trace_family = "HDiv Trace"
        T = FunctionSpace(mesh, trace_family, degree)
    W = V * T

    # Trial and test functions
    # solution = Function(W)
    # u, p, lambda_h = split(solution)
    p, lambda_h = TrialFunctions(W)
    q, mu_h  = TestFunctions(W)

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

    # Dirichlet BCs
    p_boundaries = Constant(0.0)
    bc_multiplier = DirichletBC(W.sub(1), p_exact, "on_boundary")

    # Hybridization parameter
    s = Constant(-1.0)
    beta = Constant(32.0)
    h = CellDiameter(mesh)
    h_avg = avg(h)

    # Classical term
    a = dot(grad(p), grad(q)) * dx
    L = f * q * dx
    # Hybridization terms
    a += s * dot(grad(q), n)("+") * (p("+") - lambda_h("+")) * dS
    a += -dot(grad(p), n)("+") * (q("+") - mu_h("+")) * dS
    a += (beta / h_avg) * (p("+") - lambda_h("+")) * (q("+") - mu_h("+")) * dS
    # Boundary terms
    # a += -dot(vel_projected, n) * v * ds  # How to set this bc??
    # a += (beta / h) * (p- p_boundaries) * q * ds  # is this necessary?
    L += s * dot(grad(q), n) * p_boundaries * ds

    F = a - L
    a_form = lhs(F)

    _A = Tensor(a_form)
    A = _A.blocks
    S = A[1, 1] - A[1, :1] * A[:1, :1].inv * A[:1, 1]
    Smat = assemble(S, bcs=bc_multiplier)
    petsc_mat = Smat.M.handle

    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = Mnp.shape[0]

    num_of_factors = int(0.99 * number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors, backend="slepc", is_negative_spectrum=False)

    result = ConditionNumberResult(
        form=a,
        assembled_form=Smat,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric,
        bcs=bc_multiplier
    )
    return result


def solve_poisson_lsh(
    num_elements_x, 
    num_elements_y, 
    degree=1, 
    use_quads=False,
    is_multiplier_continuous=False
):
    # Defining the mesh
    mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

    # Function space declaration
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
    # solution = Function(W)
    # u, p, lambda_h = split(solution)
    u, p, lambda_h = TrialFunctions(W)
    v, q, mu_h  = TestFunctions(W)

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

    # BCs
    bcs = DirichletBC(W.sub(2), p_exact, "on_boundary")

    # Hybridization parameter
    beta_0 = Constant(1.0)
    beta = beta_0 / h
    beta_avg = beta_0 / h("+")

    # Stabilizing parameter
    delta_1 = Constant(1)
    delta_2 = Constant(1)
    delta_3 = Constant(1)

    # Numerical flux trace
    u_hat = u + beta * (p - lambda_h) * n

    # Flux least-squares
    a = (
        (inner(u, v) - q * div(u) - p * div(v) + inner(grad(p), grad(q)))
        * delta_1
        * dx
    )
    # These terms below are unsymmetric
    a += delta_1 * jump(u_hat, n=n) * q("+") * dS
    a += delta_1 * dot(u_hat, n) * q * ds
    # a += delta_1 * dot(u, n) * q * ds
    # L = -delta_1 * dot(u_projected, n) * q * ds
    a += delta_1 * lambda_h("+") * jump(v, n=n) * dS
    a += delta_1 * lambda_h * dot(v, n) * ds
    # L = -delta_1 * p_boundaries * dot(v, n) * ds

    # Mass balance least-square
    a += delta_2 * div(u) * div(v) * dx
    # L = delta_2 * f * div(v) * dx

    # Irrotational least-squares
    a += delta_3 * inner(curl(u), curl(v)) * dx

    # Hybridization terms
    a += mu_h("+") * jump(u_hat, n=n) * dS

    # Weakly imposed BC from hybridization
    # a += mu_h * (lambda_h - p_boundaries) * ds
    # a += mu_h * lambda_h * ds
    # ###
    # a += (
    #     (mu_h - q) * (lambda_h - p_boundaries) * ds
    # )  # maybe this is not a good way to impose BC, but this necessary

    _A = Tensor(a)
    A = _A.blocks
    S = A[2, 2] - A[2, :2] * A[:2, :2].inv * A[:2, 2]
    Smat = assemble(S, bcs=bcs)
    petsc_mat = Smat.M.handle

    is_symmetric = petsc_mat.isSymmetric(tol=1e-8)
    size = petsc_mat.getSize()
    Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
    Mnp.eliminate_zeros()
    nnz = Mnp.nnz
    number_of_dofs = Mnp.shape[0]

    num_of_factors = int(0.99 * number_of_dofs) - 1
    condition_number = calculate_condition_number(petsc_mat, num_of_factors, backend="slepc", is_negative_spectrum=True)

    result = ConditionNumberResult(
        form=a,
        assembled_form=Smat,
        condition_number=condition_number,
        sparse_operator=Mnp,
        number_of_dofs=number_of_dofs,
        nnz=nnz,
        is_operator_symmetric=is_symmetric,
        bcs=bcs
    )
    return result


def hp_refinement_cond_number_calculation(
    solver,
    min_degree=1,
    max_degree=4,
    numel_xy=(5, 10, 15, 20, 25),
    quadrilateral=True,
    name="",
    **kwargs
):
    results_dict = {
        "Element": list(),
        "Number of Elements": list(),
        "Degree": list(),
        "Symmetric": list(),
        "nnz": list(),
        "dofs": list(),
        "Condition Number": list(),
    }
    element_kind = "Quad" if quadrilateral else "Tri"
    pbar = tqdm(range(min_degree, max_degree))
    for degree in pbar:
        for n in numel_xy:
            pbar.set_description(f"Processing {name} - degree = {degree} - N = {n}")
            result = solver(n, n, degree=degree, use_quads=quadrilateral)

            results_dict["Element"].append(element_kind)
            results_dict["Number of Elements"].append(n * n)
            results_dict["Degree"].append(degree)
            results_dict["Symmetric"].append(result.is_operator_symmetric)
            results_dict["nnz"].append(result.nnz)
            results_dict["dofs"].append(result.number_of_dofs)
            results_dict["Condition Number"].append(result.condition_number)

    os.makedirs("./cond_number_results/results_%s" % name, exist_ok=True)
    df_cond_number = pd.DataFrame(data=results_dict)
    path_to_save_results = "./cond_number_results/results_%s/cond_numbers.csv" % name
    df_cond_number.to_csv(path_to_save_results)

    return df_cond_number


# Solver options
solvers_options = {
    # "cg": solve_poisson_cg,
    # "cgls": solve_poisson_cgls,
    # "dgls": solve_poisson_dgls,
    # "sdhm": solve_poisson_sdhm,
    # "ls": solve_poisson_ls,
    # "dls": solve_poisson_dls,
    # "lsh": solve_poisson_lsh,  # this method implementation should be reviewed (regarding BC handling)
    # "vms": solve_poisson_vms,
    # "dvms": solve_poisson_dvms,
    # "mixed_RT": solve_poisson_mixed_RT,
    # "hdg": solve_poisson_hdg,
    # "cgh": solve_poisson_cgh,
    "ldgc": solve_poisson_ldgc,
}

degree = 1
last_degree = 1
for current_solver in solvers_options:

    # Setting the output file name
    name = f"{current_solver}"

    # Selecting the solver and its kwargs
    solver = solvers_options[current_solver]

    # Performing the convergence study
    hp_refinement_cond_number_calculation(
        solver,
        min_degree=degree,
        max_degree=degree + last_degree,
        quadrilateral=True,
        name=name
    )

# N = 5
# result = solve_poisson_ldgc(N, N, degree=1, use_quads=True)

# print(f'Is symmetric? {result.is_operator_symmetric}')
# print(f'nnz: {result.nnz}')
# print(f'DoFs: {result.number_of_dofs}')
# print(f'Condition Number: {result.condition_number}')

# # Plotting the resulting matrix
# import copy
# my_cmap = copy.copy(plt.cm.get_cmap("winter"))
# my_cmap.set_bad(color="lightgray")
# # plot_matrix_primal_hybrid_full(result.form, result.bcs, cmap=my_cmap)
# # plot_matrix_mixed_hybrid_full(result.form, result.bcs, cmap=my_cmap)
# plot_matrix_hybrid_multiplier(result.form, trace_index=1, bcs=result.bcs, cmap=my_cmap)
# # plot_matrix(result.assembled_form, cmap=my_cmap)
# # plot_matrix_mixed(result.assembled_form, cmap=my_cmap)
# plt.tight_layout()
# plt.savefig("sparse_pattern.png")
# plt.show()