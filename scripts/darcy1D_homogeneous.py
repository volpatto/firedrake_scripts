from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# Mesh definition
Lx = 1.0
numel = 50
mesh = IntervalMesh(numel, 0.0, Lx)

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)

# Essential boundary conditions
boundary_value = 0.0
bc = DirichletBC(V, boundary_value, [1, 2])  # Boundary condition in 1 and 2 marked bounds (left and right)

# Trial and Test functions
p = TrialFunction(V)
v = TestFunction(V)

# Interpolating in the permeability space
k = Constant(1.0)

# Source term
f = Constant(1.0)

# Variational form
a = inner(k * grad(p), grad(v)) * dx
L = f * v * dx

# Defining the problem
solution = Function(V)
poisson_problem = LinearVariationalProblem(a, L, solution, bcs=bc)

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

# Writing the solution as a vector
x = SpatialCoordinate(mesh)
x_values = mesh.coordinates.vector().dat.data
sol_vec = np.array(solution.vector().dat.data)
print('\ncoordinates:\n', x_values)
print('\nsolution:\n', sol_vec)
print('\ncondutivity:\n', float(k.dat.data))

# Plotting the solution
plot(solution)
plot(flux)
plot(grad_p)
plt.show()
