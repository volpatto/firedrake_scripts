import betterspy as spy
from firedrake import *
from firedrake import PETSc
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from slepc4py import SLEPc
import time

parameters["pyop2_options"]["lazy_evaluation"] = True

# Defining the mesh
num_elements_x = num_elements_y = 10
use_quads = True
mesh = UnitSquareMesh(num_elements_x, num_elements_y, quadrilateral=use_quads)

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

A = assemble(a, bcs=bcs, mat_type="aij")
petsc_mat = A.M.handle
size = petsc_mat.getSize()

time_begin_scipy = time.time()

Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)
Mnp.eliminate_zeros()  # in-place operation
nnz = Mnp.nnz

_, largest_singular_values, _ = svds(Mnp, which="LM")
_, smallest_singular_values, _ = svds(Mnp, which="SM")
zero_tol = 1e-8
smallest_singular_values = smallest_singular_values[smallest_singular_values > zero_tol]
condition_number = largest_singular_values.max() / smallest_singular_values.min()

time_scipy = time.time() - time_begin_scipy

print(f'Is symmetric? {petsc_mat.isSymmetric(tol=1e-8)}')
print(f'nnz: {nnz}')
print(f'DoFs: {W.dim()}')
print(f'Condition Number (scipy): {condition_number}')

time_begin_slepc = time.time()

S = SLEPc.SVD()
S.create()
S.setOperator(petsc_mat)
S.setType(SLEPc.SVD.Type.LAPACK)
S.setWhichSingularTriplets(which=S.Which.LARGEST)
S.setFromOptions()
S.solve()

# Create the results vectors
vr, vi = petsc_mat.getVecs()
nconv = S.getConverged()
singular_values_list = list()
num_of_extreme_singular_values = 5
if nconv > 0:
    for i in range(num_of_extreme_singular_values):
        singular_value_low = S.getSingularTriplet(i, vr, vi)
        singular_value_high = S.getSingularTriplet(nconv - 1 - i, vr, vi)
        singular_values_list.append(singular_value_low)
        singular_values_list.append(singular_value_high)

singular_values = np.array(singular_values_list)
zero_tol = 1e-8
singular_values = singular_values[singular_values > zero_tol]
condition_number_slepc = singular_values.max() / singular_values.min()

time_slepc = time.time() - time_begin_slepc

print(f"Condition Number (SLEPc): {condition_number_slepc}")

print(f'Time to calculate with SciPy: {time_scipy}')
print(f'Time to calculate with SLEPc (serial): {time_slepc}')

# spy.plot(Mnp, border_width=5, border_color="0")
# plt.show()
