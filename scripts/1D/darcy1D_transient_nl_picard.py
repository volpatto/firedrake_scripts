from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# Mesh definition
numel = 200
L = 10.0
x_left, x_right = 0.0, L
mesh = IntervalMesh(numel, x_left, x_right)

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)
Vref = FunctionSpace(mesh, "CG", 1)

# Essential boundary conditions
boundary_value_left = 5e5
bc_left = DirichletBC(V, boundary_value_left, 1)  # Boundary condition in 1 marked bounds (left)
bcs = [bc_left]

# Trial and Test functions
p = TrialFunction(V)
v = TestFunction(V)

# Source term
f = Constant(0.0)

# Initial condition
ic = Constant(3e7)

# Time parameters
T_total = 4.147e7  # 480 days
dt = T_total / 500.

# Assigning the IC
p_k = interpolate(ic, V)  # Previous time
p_p = interpolate(ic, V)  # Delayed Picard


# Compressibility factor fitted from PR-EoS in terms of pressure
def Z(p):
    zcoef = np.zeros(10)
    zcoef[0] = 9.99921144125617722409060661448165774345397949218750e-01
    zcoef[1] = -1.19919829040115086206268166821135856547897446944262e-08
    zcoef[2] = 2.95290097079410864772724180594317997860333752145959e-16
    zcoef[3] = 9.42231835327529024372629441226386885009463920848079e-24
    zcoef[4] = -2.46929568577390678055712987160325876321312377285599e-31
    zcoef[5] = -7.40016953399249021667823040632784926032215181166411e-39
    zcoef[6] = 4.21756086831535086775256556143298098377555103824524e-46
    zcoef[7] = -7.90995787006734393072263251251053413138923617742665e-54
    zcoef[8] = 6.96584174374744927048653426883823314263335677413527e-62
    zcoef[9] = -2.42926517319393920651606665298459161434867679132808e-70

    Z_value = 0.0
    for i in range(len(zcoef)):
        Z_value += zcoef[i] * p ** i

    return Z_value


# Non-linear pressure term linearized by Picard
def fp(p, p_k):
    return p / Z(p_k)


# Model parameters
phi = Constant(0.05)  # Porosity
kappa = Constant(1e-18)  # Permeability
mu = Constant(1e-5)  # Methane's viscosity

# Variational formulation
a = phi * inner(fp(p, p_p) / dt, v) * dx + (kappa / mu) * inner(fp(p_p, p_p) * grad(p), grad(v)) * dx
L = phi * inner(fp(p_k, p_k) / dt, v) * dx + f * v * dx

solver_parameters = {
    'ksp_type': 'preonly',
    'pc_type': 'lu'
}

# Picard iteration parameters
tolerance = 1e-3
norm_l2 = 1.0  # Any value greater than tolerance
it_max = 100  # Number of Picard iteration limit

# Iterating and solving over the time
t = dt
step = 0
x_values = mesh.coordinates.vector().dat.data
sol_values = []
p_values_deg1 = []
psol_deg1 = Function(Vref)
while t <= T_total:
    step += 1
    print('============================')
    print('\ttime =', t)
    print('\tstep =', step)

    # Picard iteration loop
    picard_has_converged = False
    # Load vector assembling. LHS needs to be assembled only one time. It remains constant over a time step.
    b = assemble(L)
    bc_left.apply(b)
    for i in range(1, it_max + 1):
        print(f'\tPicard iteration: {i}')
        # Stiffness matrix assembling
        A = assemble(a, bcs=bcs)
        # Defining the unknown
        p = Function(V)
        # Solving the linear system
        solve(A, p, b, solver_parameters=solver_parameters)
        norm_l2 = norm(p - p_p, mesh=mesh)
        p_p.assign(p)
        if norm_l2 < tolerance:
            picard_has_converged = True
            break
    if not picard_has_converged:
        print('*** Convergence failed ***')
        break
    sol_vec = np.array(p.vector().dat.data)
    sol_values.append(sol_vec)
    psol_deg1.project(p)
    p_vec_deg1 = np.array(psol_deg1.vector().dat.data)
    p_values_deg1.append(p_vec_deg1)
    p_k.assign(p)

    t += dt
    print('============================')

# *** Plotting ***

if picard_has_converged:
    # Setting up the figure object
    fig = plt.figure(dpi=300, figsize=(8, 6))
    ax = plt.subplot(111)

    # Plotting the data
    steps_to_plot = [1, 10, 30, 60, 120, 360, 480]
    for i in steps_to_plot:
        ax.plot(x_values, p_values_deg1[i-1] / 1e3, label=('Day %i' % (i)))

    # Getting and setting the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 1.05 * box.width, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Setting the xy-labels
    plt.xlabel(r'$x$ [m]')
    plt.ylabel(r'Pressure [kPa]')
    plt.xlim(x_values.min(), x_values.max())

    # Setting the grids in the figure
    plt.minorticks_on()
    plt.grid(True)
    plt.grid(False, linestyle='--', linewidth=0.5, which='major')
    plt.grid(False, linestyle='--', linewidth=0.1, which='minor')

    # Displaying the plot
    plt.tight_layout()
    plt.savefig('compressible-flow-picard.png')
