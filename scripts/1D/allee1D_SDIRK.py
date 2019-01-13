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

# SDIRK functions
U_1 = Function(V)
U_2 = Function(V)
U_3 = Function(V)
U_4 = Function(V)
U_5 = Function(V)
U_6 = Function(V)
u_sol_rk = Function(V)

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
    return -beta * u + (1.0 + beta) * u * u - u * u * u


def diffusion_advection(u, w):
    diffusion = inner(grad(u), grad(w))
    advection = inner(v(u) * grad(u)[0], w)
    return diffusion + advection


def spatial_operator(u, w):
    return reaction_term(u) * w - diffusion_advection(u, w)


# Velocity term
def v(u):
    return v0 + v1 * u


# Time parameters
Total_time = 160.
dt = 5.0
Dt = Constant(dt)

# *** SDIRK4 constants from Butcher tableau ***
a_coeff = np.zeros((6, 6))
b_coeff = np.zeros(6)
c_coeff = np.zeros(6)
# a values
a_coeff[1][0] = 1 / 4
a_coeff[1][1] = 1 / 4
a_coeff[2][0] = 8611 / 62500
a_coeff[2][1] = - 1743 / 31250
a_coeff[2][2] = 1 / 4
a_coeff[3][0] = 5012029 / 34652500
a_coeff[3][1] = - 654441 / 2922500
a_coeff[3][2] = 174375 / 388108
a_coeff[3][3] = 1 / 4
a_coeff[4][0] = 15267082809 / 155376265600
a_coeff[4][1] = - 71443401 / 120774400
a_coeff[4][2] = 730878875 / 902184768
a_coeff[4][3] = 2285395 / 8070912
a_coeff[4][4] = 1 / 4
a_coeff[5][0] = 82889 / 524892
a_coeff[5][2] = 15625 / 83664
a_coeff[5][3] = 69875 / 102672
a_coeff[5][4] = - 2260 / 8211
a_coeff[5][5] = 1 / 4
# b values
b_coeff[0] = 82889 / 524892
b_coeff[2] = 15625 / 83664
b_coeff[3] = 69875 / 102672
b_coeff[4] = - 2260 / 8211
b_coeff[5] = 1 / 4
# c values
c_coeff[1] = 1 / 2
c_coeff[2] = 83 / 250
c_coeff[3] = 31 / 50
c_coeff[4] = 17 / 20
c_coeff[5] = 1


def sdirk4_solver(u_k):
    # Solver parameters
    solver_parameters = {
        'mat_type': 'aij',
        'snes_tyoe': 'newtonls',
        'pc_type': 'lu'
    }
    # Stage 1
    U_1.assign(u_k)
    # Stage 2
    cumulative_term = w * U_1 * dx + Dt * (a_coeff[1][0] * spatial_operator(U_1, w) + a_coeff[1][1] * spatial_operator(U_2, w)) * dx
    A_step_2 = U_2 * w * dx - cumulative_term
    solve(A_step_2 == 0, U_2, solver_parameters=solver_parameters)
    # Stage 3
    cumulative_term = w * U_1 * dx \
                      + Dt * (a_coeff[2][0] * spatial_operator(U_1, w)
                              + a_coeff[2][1] * spatial_operator(U_2, w)
                              + a_coeff[2][2] * spatial_operator(U_3, w)) * dx
    A_step_3 = U_3 * w * dx - cumulative_term
    solve(A_step_3 == 0, U_3, solver_parameters=solver_parameters)
    # Stage 4
    cumulative_term = w * U_1 * dx \
                      + Dt * (a_coeff[3][0] * spatial_operator(U_1, w)
                              + a_coeff[3][1] * spatial_operator(U_2, w)
                              + a_coeff[3][2] * spatial_operator(U_3, w)
                              + a_coeff[3][3] * spatial_operator(U_4, w)) * dx
    A_step_4 = U_4 * w * dx - cumulative_term
    solve(A_step_4 == 0, U_4, solver_parameters=solver_parameters)
    # Stage 5
    cumulative_term = w * U_1 * dx \
                      + Dt * (a_coeff[4][0] * spatial_operator(U_1, w)
                              + a_coeff[4][1] * spatial_operator(U_2, w)
                              + a_coeff[4][2] * spatial_operator(U_3, w)
                              + a_coeff[4][3] * spatial_operator(U_4, w)
                              + a_coeff[4][4] * spatial_operator(U_5, w)) * dx
    A_step_5 = U_5 * w * dx - cumulative_term
    solve(A_step_5 == 0, U_5, solver_parameters=solver_parameters)
    # Stage 6
    cumulative_term = w * U_1 * dx \
                      + Dt * (a_coeff[5][0] * spatial_operator(U_1, w)
                              + a_coeff[5][1] * spatial_operator(U_2, w)
                              + a_coeff[5][2] * spatial_operator(U_3, w)
                              + a_coeff[5][3] * spatial_operator(U_4, w)
                              + a_coeff[5][4] * spatial_operator(U_5, w)
                              + a_coeff[5][5] * spatial_operator(U_6, w)) * dx
    A_step_6 = U_6 * w * dx - cumulative_term
    solve(A_step_6 == 0, U_6, solver_parameters=solver_parameters)

    # Updating the solution
    rhs = U_1 * w * dx + Dt * (
            Constant(b_coeff[0]) * spatial_operator(U_1, w)
            + Constant(b_coeff[1]) * spatial_operator(U_2, w)
            + Constant(b_coeff[2]) * spatial_operator(U_3, w)
            + Constant(b_coeff[3]) * spatial_operator(U_4, w)
            + Constant(b_coeff[4]) * spatial_operator(U_5, w)
            + Constant(b_coeff[5]) * spatial_operator(U_6, w)
        ) * dx
    A_sol = inner(u_sol_rk, w) * dx
    solve(A_sol - rhs == 0, u_sol_rk)
    return u_sol_rk


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

    u = sdirk4_solver(u0)
    u0.assign(u)

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
plt.savefig('alleeSDIRK.png')
