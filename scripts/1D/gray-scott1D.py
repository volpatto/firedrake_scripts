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
degree = 3  # Polynomial degree of approximation
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

# Time parameters
Total_time = 4000.
dt = 1.0
Dt = Constant(dt)

# Defining residual variational form
# ** U part **
F = inner((u - u0) / Dt, p) * dx + inner(grad(u), grad(p)) * dx + u * v * v * p * dx - A * (1.0 - u) * p * dx
# ** V part **
F += inner((v - v0) / Dt, q) * dx + inner(delta_squared * grad(v), grad(q)) * dx - u * v * v * q * dx + B * v * q * dx

# Solver parameters
solver_parameters = {
    'mat_type': 'aij',
    'snes_tyoe': 'newtonls',
    'pc_type': 'lu'
}

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
    print('============================')
    print('\ttime =', t)
    print('\tstep =', step)
    print('============================')

    # solve(F == 0, w, bcs=[u_bc, v_bc], solver_parameters=solver_parameters)
    solve(F == 0, w, solver_parameters=solver_parameters)
    w0.assign(w)

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

    t += dt

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
plt.savefig('gray-scott.png')
# plt.show()

# Colormap
fig = plt.figure(dpi=300, figsize=(8, 6))
Vplot = np.array(v_values_deg1)
p = plt.imshow(Vplot, origin="lower", aspect='auto', cmap='jet')
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('gray-scott-pattern.png')
# plt.show()
