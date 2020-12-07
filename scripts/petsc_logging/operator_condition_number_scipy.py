import copy
from firedrake import *
import numpy as np
from logging import warning
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds, ArpackNoConvergence
from scipy.sparse import csr_matrix
from slepc4py import SLEPc


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

    _, largest_singular_values, _ = svds(Mnp, which="LM")
    _, smallest_singular_values, _ = svds(Mnp, which="SM")
    zero_tol = 1e-8
    smallest_singular_values = smallest_singular_values[smallest_singular_values > zero_tol]
    condition_number = largest_singular_values.max() / smallest_singular_values.min()

    return condition_number, Mnp, number_of_dofs, nnz, is_symmetric


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

    _, largest_singular_values, _ = svds(Mnp, which="LM")
    _, smallest_singular_values, _ = svds(Mnp, which="SM")
    zero_tol = 1e-8
    smallest_singular_values = smallest_singular_values[smallest_singular_values > zero_tol]
    condition_number = largest_singular_values.max() / smallest_singular_values.min()

    return condition_number, Mnp, number_of_dofs, nnz, is_symmetric


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

    _, largest_singular_values, _ = svds(Mnp, which="LM")
    _, smallest_singular_values, _ = svds(Mnp, which="SM")
    zero_tol = 1e-8
    smallest_singular_values = smallest_singular_values[smallest_singular_values > zero_tol]
    condition_number = largest_singular_values.max() / smallest_singular_values.min()

    return condition_number, Mnp, number_of_dofs, nnz, is_symmetric


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

    try:
        _, largest_singular_values, _ = svds(Mnp, which="LM", solver="arpack")
        _, smallest_singular_values, _ = svds(Mnp, which="SM", solver="arpack")
        zero_tol = 1e-10
        smallest_singular_values = smallest_singular_values[smallest_singular_values > zero_tol]
        condition_number = largest_singular_values.max() / smallest_singular_values.min()
    except ArpackNoConvergence:
        warning("SciPy Arpack svds solver did not converged. Using SLEPc to calculate cond. number instead.")
        S = SLEPc.SVD()
        S.create()
        S.setOperator(petsc_mat)
        S.setType(SLEPc.SVD.Type.LAPACK)
        S.setWhichSingularTriplets(which=S.Which.SMALLEST)
        S.setFromOptions()
        S.solve()

        # Recovering the solution
        nconv = int(S.getConverged())
        smallest_singular_values_list = list()
        largest_singular_values_list = list()
        num_of_values_in_list = 5
        num_of_extreme_singular_values = num_of_values_in_list if num_of_values_in_list < nconv else nconv
        if nconv > 0:
            for i in range(num_of_extreme_singular_values):
                smallest_singular_values_list.append(S.getValue(i))
                largest_singular_values_list.append(S.getValue(nconv - 1 - i))

        singular_values_list = smallest_singular_values_list + largest_singular_values_list
        singular_values = np.array(singular_values_list)
        zero_tol = 1e-8
        singular_values = singular_values[singular_values > zero_tol]
        condition_number = singular_values.max() / singular_values.min()

    return condition_number, Mnp, number_of_dofs, nnz, is_symmetric


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

    _, largest_singular_values, _ = svds(Mnp, which="LM")
    _, smallest_singular_values, _ = svds(Mnp, which="SM")
    zero_tol = 1e-8
    smallest_singular_values = smallest_singular_values[smallest_singular_values > zero_tol]
    condition_number = largest_singular_values.max() / smallest_singular_values.min()

    return condition_number, Mnp, number_of_dofs, nnz, is_symmetric


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

    try:
        _, largest_singular_values, _ = svds(Mnp, which="LM", solver="arpack")
        _, smallest_singular_values, _ = svds(Mnp, which="SM", solver="arpack")
        zero_tol = 1e-10
        smallest_singular_values = smallest_singular_values[smallest_singular_values > zero_tol]
        condition_number = largest_singular_values.max() / smallest_singular_values.min()
    except ArpackNoConvergence:
        warning("SciPy Arpack svds solver did not converged. Using SLEPc to calculate cond. number instead.")
        S = SLEPc.SVD()
        S.create()
        S.setOperator(petsc_mat)
        S.setType(SLEPc.SVD.Type.LAPACK)
        S.setWhichSingularTriplets(which=S.Which.SMALLEST)
        S.setFromOptions()
        S.solve()

        # Recovering the solution
        nconv = int(S.getConverged())
        smallest_singular_values_list = list()
        largest_singular_values_list = list()
        num_of_values_in_list = 5
        num_of_extreme_singular_values = num_of_values_in_list if num_of_values_in_list < nconv else nconv
        if nconv > 0:
            for idx in range(num_of_extreme_singular_values):
                smallest_singular_values_list.append(S.getValue(idx))
                largest_singular_values_list.append(S.getValue(nconv - 1 - idx))

        singular_values_list = smallest_singular_values_list + largest_singular_values_list
        singular_values = np.array(singular_values_list)
        zero_tol = 1e-8
        singular_values = singular_values[singular_values > zero_tol]
        condition_number = singular_values.max() / singular_values.min()

    return condition_number, Mnp, number_of_dofs, nnz, is_symmetric


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

    try:
        _, largest_singular_values, _ = svds(Mnp, which="LM", solver="arpack")
        _, smallest_singular_values, _ = svds(Mnp, which="SM", solver="arpack")
        zero_tol = 1e-10
        smallest_singular_values = smallest_singular_values[smallest_singular_values > zero_tol]
        condition_number = largest_singular_values.max() / smallest_singular_values.min()
    except ArpackNoConvergence:
        warning("SciPy Arpack svds solver did not converged. Using SLEPc to calculate cond. number instead.")
        S = SLEPc.SVD()
        S.create()
        S.setOperator(petsc_mat)
        S.setType(SLEPc.SVD.Type.LAPACK)
        S.setWhichSingularTriplets(which=S.Which.SMALLEST)
        S.setFromOptions()
        S.solve()

        # Recovering the solution
        nconv = int(S.getConverged())
        smallest_singular_values_list = list()
        largest_singular_values_list = list()
        num_of_values_in_list = 5
        num_of_extreme_singular_values = num_of_values_in_list if num_of_values_in_list < nconv else nconv
        if nconv > 0:
            for idx in range(num_of_extreme_singular_values):
                smallest_singular_values_list.append(S.getValue(idx))
                largest_singular_values_list.append(S.getValue(nconv - 1 - idx))

        singular_values_list = smallest_singular_values_list + largest_singular_values_list
        singular_values = np.array(singular_values_list)
        zero_tol = 1e-8
        singular_values = singular_values[singular_values > zero_tol]
        condition_number = singular_values.max() / singular_values.min()

    return condition_number, Mnp, number_of_dofs, nnz, is_symmetric


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

    try:
        _, largest_singular_values, _ = svds(Mnp, which="LM", solver="arpack")
        _, smallest_singular_values, _ = svds(Mnp, which="SM", solver="arpack")
        zero_tol = 1e-10
        smallest_singular_values = smallest_singular_values[smallest_singular_values > zero_tol]
        condition_number = largest_singular_values.max() / smallest_singular_values.min()
    except ArpackNoConvergence:
        warning("SciPy Arpack svds solver did not converged. Using SLEPc to calculate cond. number instead.")
        S = SLEPc.SVD()
        S.create()
        S.setOperator(petsc_mat)
        S.setType(SLEPc.SVD.Type.LAPACK)
        S.setWhichSingularTriplets(which=S.Which.SMALLEST)
        S.setFromOptions()
        S.solve()

        # Recovering the solution
        nconv = int(S.getConverged())
        smallest_singular_values_list = list()
        largest_singular_values_list = list()
        num_of_values_in_list = 5
        num_of_extreme_singular_values = num_of_values_in_list if num_of_values_in_list < nconv else nconv
        if nconv > 0:
            for idx in range(num_of_extreme_singular_values):
                smallest_singular_values_list.append(S.getValue(idx))
                largest_singular_values_list.append(S.getValue(nconv - 1 - idx))

        singular_values_list = smallest_singular_values_list + largest_singular_values_list
        singular_values = np.array(singular_values_list)
        zero_tol = 1e-8
        singular_values = singular_values[singular_values > zero_tol]
        condition_number = singular_values.max() / singular_values.min()

    return condition_number, Mnp, number_of_dofs, nnz, is_symmetric


N = 5
(
    condition_number,
    sparse_matrix,
    number_of_dofs,
    nnz,
    is_symmetric,
) = solve_poisson_dls(N, N, degree=1, use_quads=True)

print(f'Is symmetric? {is_symmetric}')
print(f'nnz: {nnz}')
print(f'DoFs: {number_of_dofs}')
print(f'Condition Number: {condition_number}')

# Plotting the resulting matrix
# plt.spy(sparse_matrix, precision=1e-8)  # unclear results
Mnp = sparse_matrix.toarray()
Am = np.ma.masked_values(Mnp, 0, rtol=1e-13)
my_cmap = copy.copy(plt.cm.get_cmap("winter"))
my_cmap.set_bad(color="lightgray")

fig, ax = plt.subplots(1, 1)
ax.matshow(Am, cmap=my_cmap)
ax.tick_params(length=0)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.show()
