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
    "mat_type": "aij",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    # "mat_mumps_icntl_11": None
    "mat_mumps_icntl_4": "3",
}
u_h = Function(V)
problem = LinearVariationalProblem(a, L, u_h, bcs=bcs)
solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters)
solver.solve()

# Writing to Paraview file
outfile = File("output_poisson3D.pvd")
u_h.rename("Pressure", "label")
outfile.write(u_h, exact_solution)
