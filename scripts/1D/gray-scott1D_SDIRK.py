from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', size=14)

# Mesh definition
numel = 1000
L = 100.0
x_left, x_right = 0.0, L
mesh = IntervalMesh(numel, x_left, x_right)
x = mesh.coordinates

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)
W = MixedFunctionSpace((V, V))
Vref = FunctionSpace(mesh, "CG", 1)

# Getting trial and test functions
w = Function(W)
u, v = split(w)
p, q = TestFunction(W)

# Initial conditions
w0 = Function(W)
u0, v0 = w0.split()
u0.interpolate(1 - (1. / 2.) * sin(pi * x[0] / L) ** 100.)
v0.interpolate((1. / 4.) * sin(pi * x[0] / L) ** 100.)

# Essential boundary conditions
boundary_value_u = 1.0
boundary_value_v = 0.0
u_bc = DirichletBC(W.sub(0), boundary_value_u, [1, 2])  # Boundary condition in 1 and 2 marked bounds (left and right)
v_bc = DirichletBC(W.sub(1), boundary_value_v, [1, 2])  # Boundary condition in 1 and 2 marked bounds (left and right)

# Gray-Scott model parameters
a = 9.0
b = 0.4
delta_squared = Constant(0.01)
A = delta_squared * a
B = delta_squared ** (1. / 3.) * b


# Variational form contribution parts
def diffusion(u, v, p, q):
    diffusion_u = inner(grad(u), grad(p))
    diffusion_v = inner(delta_squared * grad(v), grad(q))
    return diffusion_u + diffusion_v


def reaction(u, v, p, q):
    reaction_u = - u * v * v * p + A * (Constant(1.0) - u) * p
    reaction_v = u * v * v * q - B * v * q
    return reaction_u + reaction_v


def spatial_operator(u, v, p, q):
    return - diffusion(u, v, p, q) + reaction(u, v, p, q)


# Time parameters
Total_time = 1000.
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


def sdirk4_solver(u0, v0):
    # SDIRK functions
    w1 = Function(W)
    # u1, v1 = split(w1)
    u1, v1 = w1.split()
    w2 = Function(W)
    u2, v2 = split(w2)
    # u2, v2 = w2.split()
    w3 = Function(W)
    u3, v3 = split(w3)
    w4 = Function(W)
    u4, v4 = split(w4)
    w5 = Function(W)
    u5, v5 = split(w5)
    w6 = Function(W)
    u6, v6 = split(w6)
    w_sol_rk = Function(W)
    u_sol_rk, v_sol_rk = split(w_sol_rk)
    # Solver parameters
    solver_parameters = {
        'mat_type': 'aij',
        'snes_tyoe': 'newtonls',
        'pc_type': 'lu'
    }
    # Stage 1
    u1 = u0
    v1 = v0
    # Stage 2
    cumulative_term = p * u1 * dx + q * v1 * dx \
                      + Dt * (a_coeff[1][0] * spatial_operator(u1, v1, p, q)
                              + a_coeff[1][1] * spatial_operator(u2, v2, p, q)) * dx
    A_step_2 = u2 * p * dx + v2 * q * dx - cumulative_term
    solve(A_step_2 == 0, w2, solver_parameters=solver_parameters)
    # Stage 3
    u2, v2 = w2.split()
    cumulative_term = p * u1 * dx + q * v1 * dx \
                      + Dt * (a_coeff[2][0] * spatial_operator(u1, v1, p, q)
                              + a_coeff[2][1] * spatial_operator(u2, v2, p, q)
                              + a_coeff[2][2] * spatial_operator(u3, v3, p, q)) * dx
    A_step_3 = u3 * p * dx + v3 * q * dx - cumulative_term
    solve(A_step_3 == 0, w3, solver_parameters=solver_parameters)
    # Stage 4
    u3, v3 = w3.split()
    cumulative_term = p * u1 * dx + q * v1 * dx \
                      + Dt * (a_coeff[3][0] * spatial_operator(u1, v1, p, q)
                              + a_coeff[3][1] * spatial_operator(u2, v2, p, q)
                              + a_coeff[3][2] * spatial_operator(u3, v3, p, q)
                              + a_coeff[3][3] * spatial_operator(u4, v4, p, q)) * dx
    A_step_4 = u4 * p * dx + v4 * q * dx - cumulative_term
    solve(A_step_4 == 0, w4, solver_parameters=solver_parameters)
    # Stage 5
    u4, v4 = w4.split()
    cumulative_term = p * u1 * dx + q * v1 * dx \
                      + Dt * (a_coeff[4][0] * spatial_operator(u1, v1, p, q)
                              + a_coeff[4][1] * spatial_operator(u2, v2, p, q)
                              + a_coeff[4][2] * spatial_operator(u3, v3, p, q)
                              + a_coeff[4][3] * spatial_operator(u4, v4, p, q)
                              + a_coeff[4][4] * spatial_operator(u5, v5, p, q)) * dx
    A_step_5 = u5 * p * dx + v5 * q * dx - cumulative_term
    solve(A_step_5 == 0, w5, solver_parameters=solver_parameters)
    # Stage 6
    u5, v5 = w5.split()
    cumulative_term = p * u1 * dx + q * v1 * dx \
                      + Dt * (a_coeff[5][0] * spatial_operator(u1, v1, p, q)
                              + a_coeff[5][1] * spatial_operator(u2, v2, p, q)
                              + a_coeff[5][2] * spatial_operator(u3, v3, p, q)
                              + a_coeff[5][3] * spatial_operator(u4, v4, p, q)
                              + a_coeff[5][4] * spatial_operator(u5, v5, p, q)
                              + a_coeff[5][5] * spatial_operator(u6, v6, p, q)) * dx
    A_step_6 = u6 * p * dx + v6 * q * dx - cumulative_term
    solve(A_step_6 == 0, w6, solver_parameters=solver_parameters)

    # Updating the solution
    u6, v6 = w6.split()
    rhs = p * u1 * dx + q * v1 * dx + Dt * (
            Constant(b_coeff[0]) * spatial_operator(u1, v1, p, q)
            + Constant(b_coeff[1]) * spatial_operator(u2, v2, p, q)
            + Constant(b_coeff[2]) * spatial_operator(u3, v3, p, q)
            + Constant(b_coeff[3]) * spatial_operator(u4, v4, p, q)
            + Constant(b_coeff[4]) * spatial_operator(u5, v5, p, q)
            + Constant(b_coeff[5]) * spatial_operator(u6, v6, p, q)
        ) * dx
    A_sol = p * u_sol_rk * dx + q * v_sol_rk * dx
    solve(A_sol - rhs == 0, w_sol_rk)
    return w_sol_rk


# Iterating and solving over the time
step = 0
t = 0.0
x_values = mesh.coordinates.vector().dat.data
u_values = []
v_values = []
u_values_deg1 = []
v_values_deg1 = []
usol_deg1 = Function(Vref)
vsol_deg1 = Function(Vref)
while t < Total_time:
    step += 1
    t += dt
    print('============================')
    print('\ttime =', t)
    print('\tstep =', step)
    print('============================')

    w = sdirk4_solver(u0, v0)
    w0.assign(w)
    u0, v0 = split(w0)

    usol, vsol = w.split()
    usol_deg1.project(usol)
    vsol_deg1.project(vsol)
    u_vec = np.array(usol.vector().dat.data)
    u_values.append(u_vec)
    u_vec_deg1 = np.array(usol_deg1.vector().dat.data)
    u_values_deg1.append(u_vec_deg1)
    v_vec = np.array(vsol.vector().dat.data)
    v_values.append(v_vec)
    v_vec_deg1 = np.array(vsol_deg1.vector().dat.data)
    v_values_deg1.append(v_vec_deg1)

# Setting up the figure object

fig = plt.figure(dpi=300, figsize=(8, 6))
ax = plt.subplot(111)

# Plotting
ax.plot(x_values, u_values_deg1[step-1], '--', label='U')
ax.plot(x_values, v_values_deg1[step-1], label='V')

# Getting and setting the legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, 1.05 * box.width, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Setting the xy-labels
plt.xlabel(r'$x$ [L]')
plt.ylabel(r'concentration [adim]')
plt.xlim(x_values.min(), x_values.max())

# Setting the grids in the figure
plt.minorticks_on()
plt.grid(True)
plt.grid(False, linestyle='--', linewidth=0.5, which='major')
plt.grid(False, linestyle='--', linewidth=0.1, which='minor')

plt.tight_layout()
plt.savefig('gray-scott_SDIRK.png')
# plt.show()

# Colormap
fig = plt.figure(dpi=300, figsize=(8, 6))
Vplot = np.array(v_values_deg1)
p = plt.imshow(Vplot, origin="lower", aspect='auto', cmap='jet')
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('gray-scott-pattern_SDIRK.png')
# plt.show()
