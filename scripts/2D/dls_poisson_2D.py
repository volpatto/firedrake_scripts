from firedrake import *
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc


def print(content_to_print):
    return PETSc.Sys.Print(content_to_print)


parameters["pyop2_options"]["lazy_evaluation"] = False
PETSc.Log.begin()

# Defining the mesh
N = 20
use_quads = True
mesh = UnitSquareMesh(N, N, quadrilateral=use_quads)
comm = mesh.comm

# Function space declaration
pressure_family = 'DQ' if use_quads else 'DG'
velocity_family = 'DQ' if use_quads else 'DG'
degree = 1
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

# Forcing function
f_expression = div(-grad(p_exact))
f = Function(V).interpolate(f_expression)

# Average cell size and mesh dependent stabilization
h_avg = (h("+") + h("-")) / 2.0

# Jump stabilizing parameters based on Badia-Codina stabilized dG method
L0 = 1
eta_p = L0 * h_avg  # method B in the Badia-Codina paper
# eta_p = 1
# eta_p = L0 * L0  # method D in the Badia-Codina paper
eta_u = h_avg / L0  # method B in the Badia-Codina paper
# eta_u = 1
# eta_u_bc = h / L0  # method B in the Badia-Codina paper
eta_u_bc = 1

# Least-Squares weights
delta = Constant(1.0e0)
# delta = h
delta_0 = delta
delta_1 = delta
delta_2 = delta
delta_3 = h * h
delta_4 = h * h

# Least-squares terms
a = delta_0 * inner(u + grad(p), v + grad(q)) * dx
a += delta_1 * div(u) * div(v) * dx
a += delta_2 * inner(curl(u), curl(v)) * dx
# Edge stabilizing terms
# ** Badia-Codina based **
# a += (eta_p / h_avg) * (jump(u, n) * jump(v, n)) * dS
# a += (eta_u / h_avg) * dot(jump(p, n), jump(q, n)) * dS
# a += (eta_u_bc / h) * p * q * ds
# ** Least-Squares based **
a += avg(delta_3) * (jump(u, n) * jump(v, n)) * dS
a += avg(delta_4) * dot(jump(p, n), jump(q, n)) * dS
a += delta_3 * p * q * ds  # may decrease convergente rates
a += delta_4 * dot(u, n) * dot(v, n) * ds  # may decrease convergente rates
# ** Mesh independent **
# a += (jump(u, n) * jump(v, n)) * dS
# a += dot(jump(p, n), jump(q, n)) * dS
# a += p * q * ds
# RHS
L = f * div(v) * dx
# Least-Squares boundary conditions
# a += (eta_p / h_avg) * dot(u, n) * dot(v, n) * ds
# L += (eta_p / h_avg) * dot(sigma_e, n) * dot(v, n) * ds
L += delta_4 * dot(sigma_e, n) * dot(v, n) * ds
# a += (eta_u / h_avg) * p * q * ds
# L += (eta_u / h_avg) * exact_solution * q * ds

# Solving the system
# solver_parameters = {
#     'ksp_monitor': None,
#     'ksp_view': None,
#     'ksp_type': 'gmres',
#     'pc_type': 'ilu',
#     'mat_type': 'aij',
#     'ksp_rtol': 1e-12,
#     'ksp_max_it': 2000
# }
solver_parameters = {
    "ksp_monitor": None,
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
solution = Function(W)
problem = LinearVariationalProblem(a, L, solution, bcs=[])
solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.snes.ksp.setConvergenceHistory()
solver.solve()

sigma_h, u_h = solution.split()
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

# Plotting velocity field exact solution
fig, axes = plt.subplots()
collection = quiver(sigma_e, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exact solution for velocity")
plt.show()

# Plotting pressure field exact solution
fig, axes = plt.subplots()
collection = tripcolor(exact_solution, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exact solution for pressure")
plt.show()

# Plotting velocity field numerical solution
fig, axes = plt.subplots()
collection = quiver(sigma_h, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Plotting pressure field numerical solution
fig, axes = plt.subplots()
collection = tripcolor(u_h, axes=axes, cmap='coolwarm')
fig.colorbar(collection)
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.show()
