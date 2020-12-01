from firedrake import *

try:
    import matplotlib.pyplot as plt

    plt.rcParams['contour.corner_mask'] = False
    plt.close('all')
except:
    warning("Matplotlib not imported")

nx, ny = 20, 20
Lx, Ly = 100.0, 100.0
len_h_x = 0.5 * Lx / nx
len_h_y = 0.5 * Ly / ny
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

plot(mesh)
plt.axis('off')

degree = 1
pressure_family = 'CG'
velocity_family = 'CG'
DG0 = FunctionSpace(mesh, 'DG', 0)
U = VectorFunctionSpace(mesh, velocity_family, degree)
V = FunctionSpace(mesh, pressure_family, degree)
W = U * V

# Trial and test functions
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
solution = Function(W)

# Mesh entities
n = FacetNormal(mesh)
h = CellDiameter(mesh)
x, y = SpatialCoordinate(mesh)

# Model parameters
k = Constant(5e-5)
mu = Constant(1.0)
rho = Constant(0.0)
g = Constant((0.0, 0.0))


# f = conditional(
#     (x - 25) * (x - 25) + (y - 50) * (y - 50) <= 100,
#     0.1,
#     conditional(
#         (x - 75) * (x - 75) + (y - 50) * (y - 50) <= 100,
#         -0.1,
#         0
#     )
# )

# class f_eval(Expression):
#     def eval(self, values, x):
#         if x[0] == 25 and x[1] == 50:
#             values[0] = 0.1
#         elif x[0] == 75 and x[1] == 50:
#             values[0] = -0.1
#         else:
#             values[0] = 0.0

# Source term
class f_eval(Expression):
    def eval(self, values, x):
        if (25 - len_h_x <= x[0] <= 25 + len_h_x) and (50 - len_h_y <= x[1] <= 50 + len_h_y):
            values[0] = 0.1
        elif (75 - len_h_x <= x[0] <= 75 + len_h_x) and (50 - len_h_y <= x[1] <= 50 + len_h_y):
            values[0] = -0.1
        else:
            values[0] = 0.0


source = project(f_eval(), DG0)

# Boundary conditions
bcs = DirichletBC(W.sub(0), Constant((0.0, 0.0)), "on_boundary")
p_boundaries = Constant(0)

# Stabilizing parameters
delta_0 = Constant(-1)
delta_1 = Constant(0.5)
delta_2 = Constant(0.0)
delta_3 = Constant(0.0)

# Mixed classical terms
a = (dot((mu / k) * u, v) - div(v) * p - delta_0 * q * div(u)) * dx
L = -delta_0 * source * q * dx - delta_0 * dot(rho * g, v) * dx - p_boundaries * dot(v, n) * ds
# Stabilizing terms
a += delta_1 * inner((k / mu) * ((mu / k) * u + grad(p)), delta_0 * (mu / k) * v + grad(q)) * dx
a += delta_2 * h * h * (mu / k) * div(u) * div(v) * dx
a += delta_3 * h * h * inner((k / mu) * curl((mu / k) * u), curl((mu / k) * v)) * dx
L += delta_1 * dot(-(k / mu) * rho * g, delta_0 * (mu / k) * v + grad(q)) * dx
L += delta_2 * h * h * (mu / k) * source * div(v) * dx

solver_parameters = {
    'ksp_type': 'lgmres',
    'pc_type': 'bjacobi',
    'mat_type': 'aij',
    'ksp_rtol': 1e-8,
    'ksp_max_it': 5000,
    'ksp_monitor_true_residual': None
}

solve(a == L, solution, bcs=bcs, solver_parameters=solver_parameters)
sigma_h, u_h = solution.split()
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

output = File('darcy_mixed_source.pvd', project_output=True)
output.write(sigma_h, u_h)

plot(sigma_h)
plot(u_h)
plt.axis('off')
plt.show()

print("\n*** DoF = %i" % W.dim())
