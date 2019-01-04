from firedrake import *
import matplotlib.pyplot as plt

# Mesh definition
numel = 10
mesh = UnitIntervalMesh(numel)

# Function space declaration
p = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", p)

# Essential boundary conditions
boundary_value = 0.0
bc = DirichletBC(V, boundary_value, [1, 2])  # Boundary condition in 1 and 2 marked bounds (left and right)

# Trial and Test functions
u = TrialFunction(V)
v = TestFunction(V)

# Source term
f = Constant(1.0)

# Variational form
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# Setting PETSc parameters and method to use a Direct Method (LU), valid for symmetric systems (be aware)
solver_parameters = {
    "ksp_type": "preonly",  # This set the method to perform only the preconditioner (LU, in the case)
    "pc_type": "lu"  # The desired preconditioner (LU)
}

solution = Function(V)
poisson_problem = LinearVariationalProblem(a, L, solution, bcs=bc)
poisson_solver = LinearVariationalSolver(poisson_problem, solver_parameters=solver_parameters)

poisson_solver.solve()

plot(solution)
plt.show()
