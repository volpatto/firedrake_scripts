from firedrake import *
import matplotlib.pyplot as plt

# Mesh definition
numel = 10
mesh = UnitIntervalMesh(numel)

# Function space declaration
p = 3  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", p)

# Essential boundary condition
boundary_val = 0.0
bc = DirichletBC(V, boundary_val, [1, 2])  # Boundary condition

# Trial and Test functions
u = TrialFunction(V)
v = TestFunction(V)

# Source term
f = Constant(1.0)

# Variational form
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# Stiffness matrix assembling
A = assemble(a, bcs=bc)

# Load vector assembling
b = assemble(L)

# Applying the essential boundary conditions to load vector
bc.apply(b)

# Printing the stiffness matrix entries and plotting
A_entries = A.M.values
plt.spy(A_entries)
plt.show()

# Solving the resultant linear problem Au = b
u = Function(V)  # Declaring the unknown as a function in the V space
solve(A, u, b)

# Plotting the solution
plot(u)
plt.show()
