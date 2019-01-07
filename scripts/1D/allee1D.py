from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', size=14)

# Mesh definition
numel = 400
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
w = TestFunction(V)

# ADR-Allee model parameters
v0 = Constant(0.0)
v1 = Constant(0.0)
beta = Constant(0.2)


# class InitialCondition(Expression):  # TODO: fix this, don't work
#     def eval(self, values, x):
#         lambda_1 = Expression(beta / sqrt(Constant(2.0)))
#         lambda_2 = Expression(1.0 / sqrt(Constant(2.0)))
#         phi_1 = Constant(100)
#         phi_2 = Constant(-100)
#         xi1 = Expression(x[0] + phi_1)
#         xi2 = Expression(x[0] + phi_2)
#         values[0] = (beta * exp(lambda_1 * xi1) + exp(lambda_2 * xi2)) / (1.0 + exp(lambda_1 * xi1) + exp(lambda_2 * xi2))

class InitialCondition(Expression):
    def eval(self, values, x):
        if x[0] < -50.0:
            values[0] = 0.0
        elif x[0] > 50.0:
            values[0] = 1
        else:
            values[0] = x[0] / 100. + 0.5


u0 = interpolate(InitialCondition(), V)


# Non-linear reaction term function
def reaction_term(u):
    return beta * u - (1.0 + beta) * u * u + u * u * u


# Time parameters
Total_time = 240.
dt = 1.0
Dt = Constant(dt)

# *** Defining residual variational form
# Temporal advection-diffusion part
F = inner((u - u0) / Dt, w) * dx + inner(grad(u), grad(w)) * dx + inner((v0 + v1 * u) * grad(u)[0], w) * dx
# Reaction part
F += inner(reaction_term(u), w) * dx

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
u_values = []
u_values_deg1 = []
usol_deg1 = Function(Vref)
# Appending initial condition to solution data
usol_deg1.project(u0)
u_vec = np.array(u0.vector().dat.data)
u_values.append(u_vec)
u_vec_deg1 = np.array(usol_deg1.vector().dat.data)
u_values_deg1.append(u_vec_deg1)
while t < Total_time:
    step += 1
    print('============================')
    print('\ttime =', t)
    print('\tstep =', step)
    print('============================')

    solve(F == 0, u, solver_parameters=solver_parameters)
    u0.assign(u)

    usol_deg1.project(u)
    u_vec = np.array(u.vector().dat.data)
    u_values.append(u_vec)
    u_vec_deg1 = np.array(usol_deg1.vector().dat.data)
    u_values_deg1.append(u_vec_deg1)

    t += dt

# Setting up the figure object

fig = plt.figure(dpi=300, figsize=(8, 6))
ax = plt.subplot(111)

# Plotting
ax.plot(x_values, u_values_deg1[0], label='0')
ax.plot(x_values, u_values_deg1[40], label='40')
ax.plot(x_values, u_values_deg1[80], label='80')
ax.plot(x_values, u_values_deg1[120], label='120')
ax.plot(x_values, u_values_deg1[160], label='160')
ax.plot(x_values, u_values_deg1[200], label='200')
ax.plot(x_values, u_values_deg1[step], label='240')

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
plt.savefig('allee.png')
# plt.show()
