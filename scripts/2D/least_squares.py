from firedrake import *
try:
    import matplotlib.pyplot as plt
    plt.rcParams['contour.corner_mask'] = False
    plt.close('all')
except:
    warning("Matplotlib not imported")

nx, ny = 10, 10
Lx, Ly = 1.0, 1.0
quadrilateral = True
mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=quadrilateral)

triplot(mesh)
plt.axis('off')

degree = 1
pressure_family = 'CG'
velocity_family = 'CG'
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
quiver(sigma_e)
source_expr = div(-(k / mu) * grad(p_exact))
f = Function(V).interpolate(source_expr)
tripcolor(sol_exact)
plt.axis('off')
plt.show()

# Boundaries: Left (1), Right (2), Bottom(3), Top (4)
vx = -2 * pi / Lx * cos(2 * pi * x / Lx) * sin(2 * pi * y / Ly)
vy = -2 * pi / Ly * sin(2 * pi * x / Lx) * cos(2 * pi * y / Ly)
p_boundaries = p_exact

bc1 = DirichletBC(W[0], as_vector([vx, 0.0]), 1)
bc2 = DirichletBC(W[0], as_vector([vx, 0.0]), 2)
bc3 = DirichletBC(W[0], as_vector([0.0, vy]), 3)
bc4 = DirichletBC(W[0], as_vector([0.0, vy]), 4)
bcs = [bc1, bc2, bc3, bc4]

# Least-squares terms
a = inner((mu / k) * u + grad(p), v + (k / mu) * grad(q)) * dx
a += div(u) * div(v) * dx
a += inner(curl((mu / k) * u), curl((mu / k) * v)) * dx
L = f * div(v) * dx

# Stiffness matrix assembling
A = assemble(a, bcs=bcs, mat_type='aij')

# Printing the stiffness matrix entries and plotting
A_entries = A.M.values
plt.spy(A_entries)
plt.axis('off')
plt.show()

solver_parameters = {
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

solve(a == L, solution, bcs=bcs, solver_parameters=solver_parameters)
sigma_h, u_h = solution.split()
sigma_h.rename('Velocity', 'label')
u_h.rename('Pressure', 'label')

output = File('ls_paper.pvd', project_output=True)
output.write(sigma_h, u_h, sol_exact, sigma_e)

quiver(sigma_h)
tripcolor(u_h)
plt.axis('off')
plt.show()

print("\n*** DoF = %i" % W.dim())
