from firedrake import *
import matplotlib.pyplot as plt

# Mesh definition
Lx = 1.0
numel = 50
mesh = IntervalMesh(numel, 0.0, Lx)

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)

# Essential boundary conditions
boundary_values = [Constant(0.0), Constant(1.0)]
bc_left = DirichletBC(V, boundary_values[0], 1)  # Boundary condition in 1 marked bound (left)
bc_right = DirichletBC(V, boundary_values[1], 2)  # Boundary condition in 2 marked bound (right)

# Trial and Test functions
p = TrialFunction(V)
v = TestFunction(V)

# Defining the space to put the permeability
kSpace = FunctionSpace(mesh, "DG", 0)

# Discontinuous permeability definition
k1 = 1.0
k2 = 0.1

# The discontinuous permeability definition
x, = SpatialCoordinate(mesh)
discontinous_permeability = conditional(le(x, Lx / 2), k1, k2)

# Interpolating in the permeability space
k = interpolate(discontinous_permeability, kSpace)

# Source term
f = Constant(1.0)

# Variational form
a = inner(k * grad(p), grad(v)) * dx
L = f * v * dx

# Defining the problem
solution = Function(V)
poisson_problem = LinearVariationalProblem(a, L, solution, bcs=[bc_left, bc_right])

# Setting PETSc parameters and method to use a Direct Method (LU), valid for symmetric systems (be aware)
solver_parameters = {
    "ksp_type": "preonly",  # This set the method to perform only the preconditioner (LU, in the case)
    "pc_type": "lu"  # The desired preconditioner (LU)
}

# Defining the solver and solving
poisson_solver = LinearVariationalSolver(poisson_problem, solver_parameters=solver_parameters)
poisson_solver.solve()

# Flux post-processing
flux = Function(V, name='Velocity')
grad_p = Function(V, name='Grad p')
flux.project(-k * grad(solution)[0])
grad_p.project(grad(solution)[0])

# Plotting the solution
plot(solution)
plt.savefig("darcy1D_heterogeneous_pressure.png")

plot(flux)
plt.savefig("darcy1D_heterogeneous_flux.png")

plot(grad_p)
plt.savefig("darcy1D_heterogeneous_grad_p.png")
# plt.show()
