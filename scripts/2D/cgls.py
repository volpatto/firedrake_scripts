from firedrake import *
try:
    import matplotlib.pyplot as plt
    plt.rcParams['contour.corner_mask'] = False
    plt.close('all')
except:
    warning("Matplotlib not imported")

nx, ny = 20, 20
Lx, Ly = 1.0, 1.0
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

plot(mesh)
plt.axis('off')

degree = 1
pressure_family = 'CG'
velocity_family = 'CG'
U = VectorFunctionSpace(mesh, velocity_family, degree)
V = FunctionSpace(mesh, pressure_family, degree + 1)
W = U * V

# Trial and test functions
u, p = TrialFunctions(W)
v, q = TestFunctions(W)
solution = Function(W)

# Mesh entities
n = FacetNormal(mesh)
x, y = SpatialCoordinate(mesh)

# Model parameters
k = Constant(1.0)
mu = Constant(1.0)
rho = Constant(0.0)
g = Constant((0.0, 0.0))

# Exact solution and source term projection
p_exact = sin(2 * pi * x / Lx) * sin(2 * pi * y / Ly)
sol_exact = Function(V).interpolate(p_exact)
sol_exact.rename('Exact pressure', 'label')
sigma_e = Function(U, name='Exact velocity')
sigma_e.project(-(k / mu) * grad(p_exact))
plot(sigma_e)
source_expr = div(-(k / mu) * grad(p_exact))
f = Function(V).interpolate(source_expr)
plot(sol_exact)
plt.axis('off')

# Boundaries: Left (1), Right (2), Bottom(3), Top (4)
vx = -2 * pi / Lx * cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly)
vy = -2 * pi / Ly * sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly)
p_boundaries = Constant(0.0)

bc1 = DirichletBC(W[0], as_vector([vx, 0.0]), 1)
bc2 = DirichletBC(W[0], as_vector([vx, 0.0]), 2)
bc3 = DirichletBC(W[0], as_vector([0.0, vy]), 3)
bc4 = DirichletBC(W[0], as_vector([0.0, vy]), 4)
bcs = [bc1, bc2, bc3, bc4]

# Mixed classical terms
a = (dot((mu / k) * u, v) - div(v) * p - q * div(u)) * dx
L = -f * q * dx - dot(rho * g, v) * dx - p_boundaries * dot(v, n) * (ds(1) + ds(2) + ds(3) + ds(4))
# Stabilizing terms
a += -0.5 * inner((k / mu) * ((mu / k) * u + grad(p)), - (mu / k) * v + grad(q)) * dx
a += 0.5 * (mu / k) * div(u) * div(v) * dx
a += 0.5 * inner((k / mu) * curl((mu / k) * u), curl((mu / k) * v)) * dx
L += 0.5 * (mu / k) * f * div(v) * dx

solver_parameters = {
    # 'ksp_type': 'tfqmr',
    'ksp_type': 'gmres',
    'pc_type': 'bjacobi',
    'mat_type': 'aij',
    'ksp_rtol': 1e-3,
    'ksp_max_it': 2000,
    'ksp_monitor': False
}

solve(a == L, solution, bcs=bcs, solver_parameters=solver_parameters)
sigma_h, u_h = solution.split()
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

output = File('cgls_paper.pvd', project_output=True)
output.write(sigma_h, u_h, sol_exact, sigma_e)

plot(sigma_h)
plot(u_h)
plt.axis('off')
plt.show()

print("\n*** DoF = %i" % W.dim())
