from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', size=14)

# Mesh definition
numel = 200
L = 250.0
x_left, x_right = -L, L
mesh = IntervalMesh(numel, x_left, x_right)
x = mesh.coordinates

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)
Vref = FunctionSpace(mesh, "CG", 1)

# Trial and Test functions
u = Function(V)
u_k = Function(V)
u_k_1 = Function(V)
u_k_2 = Function(V)
w = TestFunction(V)

# ADR-Allee model parameters
v0 = Constant(0.0)
v1 = Constant(0.0)
beta = Constant(0.2)


# Exact solution
def u_exact(t):
    lambda_1 = beta / sqrt(Constant(2.0))
    lambda_2 = 1.0 / sqrt(Constant(2.0))
    n_1 = sqrt(Constant(2.0)) * (1 + beta) - 3 * lambda_1
    n_2 = sqrt(Constant(2.0)) * (1 + beta) - 3 * lambda_2
    phi_1 = Constant(100)
    phi_2 = Constant(-100)
    xi1 = x[0] - n_1 * t + phi_1
    xi2 = x[0] - n_2 * t + phi_2
    expr = (beta * exp(lambda_1 * xi1) + exp(lambda_2 * xi2)) / (1.0 + exp(lambda_1 * xi1) + exp(lambda_2 * xi2))
    return expr


# Setting Initial Condition
expr = u_exact(0.0)
u0 = interpolate(expr, V)
u_k_1.assign(u0)


# Non-linear reaction term function
def reaction_term(u):
    return beta * u - (1.0 + beta) * u * u + u * u * u


# Velocity term
def v(u):
    return v0 + v1 * u


# Time parameters
Total_time = 160.
dt = 1.0
Dt = Constant(dt)
theta = Constant(1.0 / 2.0)

# *** Defining residual variational form with Crank Nicolson method for time discretization
# Forward temporal diffusion part
a = inner(u, w) * dx + (theta * Dt) * inner(grad(u), grad(w)) * dx
# Forward advection part
a += (theta * Dt) * inner(v(u) * grad(u)[0], w) * dx
# Forward reaction part
a += (theta * Dt) * inner(reaction_term(u), w) * dx
# Backward temporal diffusion part
L = inner(u0, w) * dx - (theta * Dt) * inner(grad(u0), grad(w)) * dx
# Backward advection part
L -= (theta * Dt) * inner(v(u0) * grad(u0)[0], w) * dx
# Backward reaction part
L -= (theta * Dt) * inner(reaction_term(u0), w) * dx

F = a - L

# Residual variational form with BDF2
f_rhs_to_lhs = - inner(grad(u), grad(w)) * dx - inner(v(u) * grad(u)[0], w) * dx - inner(reaction_term(u), w) * dx
F_BDF = inner((u - Constant(4. / 3.) * u_k_1 + Constant(1. / 3.) * u_k_2), w) * dx \
    - Constant(2. / 3.) * Dt * f_rhs_to_lhs

# Solver parameters
solver_parameters = {
    'mat_type': 'aij',
    'snes_tyoe': 'newtonls',
    'pc_type': 'lu'
}

# *** Iterating and solving over the time
step = 0
t = 0.0
x_values = mesh.coordinates.vector().dat.data
u_values = {}
u_e_values = {}
u_values_deg1 = {}
usol_deg1 = Function(Vref)
# Appending initial condition to numerical solution data
usol_deg1.project(u0)
u_vec = np.array(u0.vector().dat.data)
u_values[step] = u_vec
u_vec_deg1 = np.array(usol_deg1.vector().dat.data)
u_values_deg1[step] = u_vec_deg1
# Appending initial condition to exact solution data
expr_e = u_exact(t)
u_e = interpolate(expr_e, Vref)
u_e_vec = np.array(u_e.vector().dat.data)
u_e_values[step] = u_e_vec
# The steps we save to plot
time_factor = 1.0 / dt
steps_to_plot = [0, int(time_factor * 40), int(time_factor * 80), int(time_factor * 120), int(time_factor * 160)]
while t < Total_time:
    step += 1
    t += dt
    print('============================')
    print(f'\ttime = {t}')
    print(f'\tstep = {step}')
    print('============================')

    if step <= 2:
        solve(F == 0, u, solver_parameters=solver_parameters)
        u0.assign(u)
        u_k_2.assign(u_k_1)
        u_k_1.assign(u)
    else:
        solve(F_BDF == 0, u, solver_parameters=solver_parameters)
        u_k_2.assign(u_k_1)
        u_k_1.assign(u)

    if step in steps_to_plot:
        usol_deg1.project(u)
        u_vec = np.array(u.vector().dat.data)
        u_values[step] = u_vec
        u_vec_deg1 = np.array(usol_deg1.vector().dat.data)
        u_values_deg1[step] = u_vec_deg1
        expr_e = u_exact(t)
        u_e = interpolate(expr_e, Vref)
        u_e_vec = np.array(u_e.vector().dat.data)
        u_e_values[step] = u_e_vec

# Setting up the figure object
fig = plt.figure(dpi=300, figsize=(8, 6))
ax = plt.subplot(111)

# Plotting
for index in u_e_values:
    idx_label = int(index * dt)
    ax.plot(x_values, u_values_deg1[index], label=f'{idx_label}')
    ax.plot(x_values, u_e_values[index], '--', label='exact')

# Getting and setting the legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, 1.05 * box.width, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Setting the xy-labels
plt.xlabel(r'space $(x)$')
plt.ylabel(r'Population Density $(u)$')
plt.xlim(x_values.min(), x_values.max())

# Setting the grids in the figure
plt.minorticks_on()
plt.grid(True)
plt.grid(False, linestyle='--', linewidth=0.5, which='major')
plt.grid(False, linestyle='--', linewidth=0.1, which='minor')

plt.tight_layout()
plt.savefig('alleeBDF.png')
