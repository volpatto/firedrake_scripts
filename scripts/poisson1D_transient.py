from firedrake import *
import matplotlib.pyplot as plt

# Mesh definition
numel = 20
mesh = UnitIntervalMesh(numel)

# Function space declaration
p = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", p)

# Essential boundary conditions
boundary_value = 0.0
bcs = DirichletBC(V, boundary_value, [1, 2])  # Boundary condition in 1 and 2 marked bounds (left and right)

# Trial and Test functions
u = Function(V)
u_k = Function(V)
v = TestFunction(V)

# Source term
f = Constant(1.0)

# Initial condition
x = SpatialCoordinate(mesh)
expr = x[0] * x[0]  # An expression to the initial condition
ic = Function(V).interpolate(expr)

# Time step
dt = 0.001

# Assigning the IC
u_k.assign(ic)
u.assign(ic)

# Residual variational formulation
F = inner((u - u_k) / dt, v) * dx + inner(grad(u), grad(v)) * dx
F -= f * v * dx

# Convergence criteria
norm_l2 = 1.0  # Any arbitrary value greater than the tolerance
tolerance = 1.e-5

# Iterating and solving over the time
t = dt
T_total = 1.0
step = 0
while t < T_total and norm_l2 > tolerance:
    step += 1
    print('============================')
    print('\ttime =', t)
    print('\tstep =', step)
    print('============================')

    solve(F == 0, u, bcs=bcs)
    norm_l2 = norm(u - u_k, mesh=mesh)
    u_k.assign(u)

    t += dt

    if step % 10 == 0:
        plot(u)

plt.show()
