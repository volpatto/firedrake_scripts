from firedrake import *
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix

parameters["pyop2_options"]["lazy_evaluation"] = False

# Defining the mesh
num_elements_x = num_elements_y = 25
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
Mnp = csr_matrix(petsc_mat.getValuesCSR()[::-1], shape=size)

_, largest_singular_values, _ = svds(Mnp, which="LM")
_, smallest_singular_values, _ = svds(Mnp, which="SM")
smallest_singular_values = smallest_singular_values[smallest_singular_values > 1e-10]
condition_number = largest_singular_values.max() / smallest_singular_values.min()

print(f'Condition Number: {condition_number}')
print(f'Is symmetric? {petsc_mat.isSymmetric(tol=1e-8)}')
