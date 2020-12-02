from firedrake import *
import numpy as np
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

    # Boundaries: Left (1), Right (2), Bottom(3), Top (4)
    vx = -2 * pi * cos(2 * pi * x) * sin(2 * pi * y)
    vy = -2 * pi * sin(2 * pi * x) * cos(2 * pi * y)

    bc1 = DirichletBC(W[0], as_vector([vx, 0.0]), 1)
    bc2 = DirichletBC(W[0], as_vector([vx, 0.0]), 2)
    bc3 = DirichletBC(W[0], as_vector([0.0, vy]), 3)
    bc4 = DirichletBC(W[0], as_vector([0.0, vy]), 4)
    bcs = [bc1, bc2, bc3, bc4]

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

    # Boundaries: Left (1), Right (2), Bottom(3), Top (4)
    vx = -2 * pi * cos(2 * pi * x) * sin(2 * pi * y)
    vy = -2 * pi * sin(2 * pi * x) * cos(2 * pi * y)

    bc1 = DirichletBC(W[0], as_vector([vx, 0.0]), 1)
    bc2 = DirichletBC(W[0], as_vector([vx, 0.0]), 2)
    bc3 = DirichletBC(W[0], as_vector([0.0, vy]), 3)
    bc4 = DirichletBC(W[0], as_vector([0.0, vy]), 4)
    bcs = [bc1, bc2, bc3, bc4]

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

    # Boundaries: Left (1), Right (2), Bottom(3), Top (4)
    vx = -2 * pi * cos(2 * pi * x) * sin(2 * pi * y)
    vy = -2 * pi * sin(2 * pi * x) * cos(2 * pi * y)

    bc1 = DirichletBC(W[0], as_vector([vx, 0.0]), 1)
    bc2 = DirichletBC(W[0], as_vector([vx, 0.0]), 2)
    bc3 = DirichletBC(W[0], as_vector([0.0, vy]), 3)
    bc4 = DirichletBC(W[0], as_vector([0.0, vy]), 4)
    bcs = [bc1, bc2, bc3, bc4]

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


N = 10
condition_number, sparse_matrix, number_of_dofs, nnz, is_symmetric = solve_poisson_cgls(N, N, degree=1, use_quads=True)

print(f'Is symmetric? {is_symmetric}')
print(f'nnz: {nnz}')
print(f'DoFs: {number_of_dofs}')
print(f'Condition Number: {condition_number}')

plt.spy(sparse_matrix, precision=1e-8)
plt.show()
