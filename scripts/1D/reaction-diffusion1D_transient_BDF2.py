from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', size=14)

# Mesh definition
numel = 200
x_left, x_right = -1.0, 1.0
mesh = IntervalMesh(numel, x_left, x_right)

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)

# Essential boundary conditions
boundary_value = 0.0
bcs = DirichletBC(V, boundary_value, [1, 2])  # Boundary condition in 1 and 2 marked bounds (left and right)

# Trial and Test functions
u = Function(V)
u_k = Function(V)
u_k_1 = Function(V)
u_k_2 = Function(V)
v = TestFunction(V)

# Source term
f = Constant(0.0)

# Growth rate parameter
r = Constant(0.3)

# Diffusion parameter
D = Constant(1.e0)

# Initial condition
x = SpatialCoordinate(mesh)
expr = exp(- 200.0 * x[0] * x[0])  # An expression to the initial condition
ic = Function(V).interpolate(expr)

# Time step
dt = 0.1

# Assigning the IC
u_k.assign(ic)
u_k_1.assign(ic)

# Residual variational formulation with implicit Euler for time
F = inner((u - u_k) / dt, v) * dx + inner(D * grad(u), grad(v)) * dx - inner(r * u, v) * dx
F -= f * v * dx

# Residual variational form with BDF2
f_rhs_to_lhs = - inner(D * grad(u), grad(v)) * dx + inner(r * u, v) * dx
F_BDF = inner((u - Constant(4. / 3.) * u_k_1 + Constant(1. / 3.) * u_k_2), v) * dx \
    - Constant(2. / 3.) * dt * f_rhs_to_lhs
F_BDF -= Constant(2. / 3.) * dt * f * v * dx

# Convergence criteria
norm_l2 = 1.0  # Any arbitrary value greater than the tolerance
tolerance = 1.e-5

# Setting PETSc parameters and method to use a Direct Method (LU), valid for symmetric systems (be aware)
solver_parameters = {
    "ksp_type": "preonly",  # This set the method to perform only the preconditioner (LU, in the case)
    "pc_type": "lu"  # The desired preconditioner (LU)
}

# Iterating and solving over the time
t = dt
T_total = 1.0
step = 0
plot_step_mod = 1
x_values = mesh.coordinates.vector().dat.data
sol_values = []
while t < T_total and norm_l2 > tolerance:
    step += 1
    print('============================')
    print('\ttime =', t)
    print('\tstep =', step)
    print('============================')

    if step <= 2:
        solve(F == 0, u, bcs=bcs, solver_parameters=solver_parameters)
        norm_l2 = norm(u - u_k, mesh=mesh)
        sol_vec = np.array(u.vector().dat.data)
        sol_values.append(sol_vec)
        u_k.assign(u)
        u_k_2.assign(u_k_1)
        u_k_1.assign(u)
    else:
        solve(F_BDF == 0, u, bcs=bcs, solver_parameters=solver_parameters)
        norm_l2 = norm(u - u_k_1, mesh=mesh)
        sol_vec = np.array(u.vector().dat.data)
        sol_values.append(sol_vec)
        u_k_2.assign(u_k_1)
        u_k_1.assign(u)

    t += dt

# *** Plotting ***

# Setting up the figure object
fig = plt.figure(dpi=300, figsize=(8, 6))
ax = plt.subplot(111)

# Plotting the data
for i in range(len(sol_values)):
    ax.plot(x_values, sol_values[i], label=('step %i' % (i + 1)))

# Getting and setting the legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, 1.01 * box.width, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Setting the xy-labels
plt.xlabel(r'$x$ [L]')
plt.ylabel(r'$u$ [population density]')
plt.xlim(x_values.min(), x_values.max())

# Setting the grids in the figure
plt.minorticks_on()
plt.grid(True)
plt.grid(False, linestyle='--', linewidth=0.5, which='major')
plt.grid(False, linestyle='--', linewidth=0.1, which='minor')

# Displaying the plot
plt.show()
