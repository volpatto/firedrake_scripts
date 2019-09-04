import time
import os
import math
from firedrake import *
import matplotlib.pyplot as plt

# get file name
fileName = os.path.splitext(__file__)[0]

# Parameters
Pe = Constant(1e10)
t_end = 10
dt = 0.5

# Create mesh and define function space
mesh = UnitSquareMesh(40, 40)
x, y = SpatialCoordinate(mesh)

# Define function spaces
V = FunctionSpace(mesh, "CG", 1)
VelSpace = VectorFunctionSpace(mesh, "CG", 2)

ic = conditional(lt((x - 0.3) * (x - 0.3) + (y - 0.3) * (y - 0.3), 0.2 ** 2), 1, 0)

b_vector = as_vector((-(y - 0.5), x - 0.5))
b = Function(VelSpace)
b.interpolate(b_vector)

bc = DirichletBC(V, Constant(0.0), "on_boundary")

# Define unknown and test function(s)
v = TestFunction(V)
u = TrialFunction(V)

u0 = Function(V)
u0.interpolate(ic)

# Stabilization
h = CellDiameter(mesh)
n = FacetNormal(mesh)
theta = Constant(1.0)

nb = sqrt(inner(b, b))
tau = 0.5 * h * pow(4.0 / (Pe * h) + 2.0 * nb, -1.0)

# first alternative: redefine the test function
# v = v + tau * inner(b, grad(v))

# second alternative: explicitly write the additional terms
r = ((1 / dt) * (u - u0) + theta * ((1.0 / Pe) * div(grad(u)) + inner(b, grad(u))) + (1 - theta) * (
            (1.0 / Pe) * div(grad(u0)) + inner(b, grad(u0)))) * tau * inner(b, grad(v)) * dx

# Define variational forms
a0 = (1.0 / Pe) * inner(grad(u0), grad(v)) * dx + inner(b, grad(u0)) * v * dx
a1 = (1.0 / Pe) * inner(grad(u), grad(v)) * dx + inner(b, grad(u)) * v * dx

A = (1 / dt) * inner(u, v) * dx - (1 / dt) * inner(u0, v) * dx + theta * a1 + (1 - theta) * a0

F = A + r

# Create files for storing results
ufile = File("results_%s/u.pvd" % fileName)

u = Function(V)
problem = LinearVariationalProblem(lhs(F), rhs(F), u, bcs=[bc])
solver = LinearVariationalSolver(problem)

u.assign(u0)

# Time-stepping
t = 0.0
ufile.write(u, time=t)

while t < t_end:
    print("t =", t, "t_end =", t_end)

    # Compute
    solver.solve()
    plot(u)

    # Move to next time step
    u0.assign(u)
    t += dt

    # Writing time step solution
    ufile.write(u, time=0)

plt.show()
