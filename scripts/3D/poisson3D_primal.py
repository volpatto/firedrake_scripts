from firedrake import *

# Defining the mesh
N = 10
mesh = UnitCubeMesh(N, N, N)

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)
U = VectorFunctionSpace(mesh, "CG", degree)

# Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)

# Mesh coordinates
x, y, z = SpatialCoordinate(mesh)

# Exact solution
p_exact = sin(2 * pi * x) * sin(2 * pi * y) * sin(2 * pi * z)
exact_solution = Function(V).interpolate(p_exact)
exact_solution.rename("Exact pressure", "label")
exact_velocity = Function(U, name="Exact velocity").project(-grad(p_exact))

# Forcing function
f_expression = div(-grad(p_exact))
f = Function(V).interpolate(f_expression)

# Dirichlet BCs
bcs = DirichletBC(V, 0.0, "on_boundary")

# Variational form
a = inner(grad(u), grad(v)) * dx
L = f * v * dx

# Solving the system
solver_parameters = {
    # In this MWE, let use HYPRE solver
    'ksp_type': 'cg',
    'pc_type': 'hypre',
    'pc_hypre_type': 'boomeramg',
    'pc_hypre_boomeramg_strong_threshold': 0.75,
    'pc_hypre_boomeramg_agg_nl': 2,
    'ksp_rtol': 1e-6,
    'ksp_atol': 1e-15
}
u_h = Function(V)
problem = LinearVariationalProblem(a, L, u_h, bcs=bcs)
solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.solve()

# Writing to Paraview file
outfile = File("output_poisson3D.pvd")
u_h.rename("Pressure", "label")
outfile.write(u_h, exact_solution)
