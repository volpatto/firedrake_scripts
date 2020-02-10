"""
This module runs a convergence history for a hybridized-DG
discretization of a model elliptic problem (detailed in the main
function). The method used is the LDG-H method.

Solves the Dirichlet problem for the elliptic equation:

-div(grad(u)) = f in [0, 1]^2, u = g on the domain boundary.

The source function f and g are chosen such that the analytic
solution is:

u(x, y) = sin(x*pi)*sin(y*pi).

This problem was crafted so that we can test the theoretical
convergence rates for the hybridized DG method: LDG-H. This
is accomplished by introducing the numerical fluxes:

u_hat = lambda,
q_hat = q + tau*(u - u_hat).

The Slate DLS in Firedrake is used to perform the static condensation
of the full LDG-H formulation of the Poisson problem to a single
system for the trace u_hat (lambda) on the mesh skeleton:

S * Lambda = E.

The resulting linear system is solved via a direct method (LU) to
ensure an accurate approximation to the trace variable. Once
the trace is solved, the Slate DSL is used again to solve the
elemental systems for the scalar solution u and its flux q.

Post-processing of the scalar variable, as well as its flux, is
performed using Slate to form and solve the elemental-systems for
new approximations u*, q*. Depending on the choice of tau, these
new solutions have superconvergent properties.

The post-processed scalar u* superconverges at a rate of k+2 when
two conditions are satisfied:

(1) q converges at a rate of k+1, and
(2) the cell average of u, ubar, superconverges at a rate of k+2.

The choice of tau heavily influences these two conditions. For all
tau > 0, the post-processed flux q* has enhanced convervation
properties! The new solution q* has the following three properties:

(1) q* converges at the same rate as q. However,
(2) q* is in H(Div), meaning that the interior jump of q* is zero!
    And lastly,
(3) div(q - q*) converges at a rate of k+1.

The expected (theoretical) rates for the LDG-H method are
summarized below for various orders of tau:

-----------------------------------------------------------------
                      u     q    ubar    u*    q*     div(p*)
-----------------------------------------------------------------
tau = O(1) (k>0)     k+1   k+1    k+2   k+2   k+1       k+1
tau = O(h) (k>0)      k    k+1    k+2   k+2   k+1       k+1
tau = O(1/h) (k>0)   k+1    k     k+1   k+1    k        k+1
-----------------------------------------------------------------

Note that the post-processing used for the flux q only holds for
simplices (triangles and tetrahedra). If someone knows of a local
post-processing method valid for quadrilaterals, please contact me!
For these numerical results, we chose the following values of tau:

tau = O(1) -> tau = 1,
tau = O(h) -> tau = h,
tau = O(1/h) -> tau = 1/h,

where h here denotes the facet area.

This demo was written by: Thomas H. Gibson (t.gibson15@imperial.ac.uk).
Modifications was done by Diego Volpatto (volpatto@lncc.br).
"""

from firedrake import *
from firedrake.petsc import PETSc
from firedrake import COMM_WORLD
import numpy as np
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", size=12)


def analytical_solution(mesh, degree):
    """
    Analytical solutions for u and q
    """

    x = SpatialCoordinate(mesh)
    V_a = FunctionSpace(mesh, "DG", degree + 3)
    U_a = VectorFunctionSpace(mesh, "DG", degree + 3)

    u_a = Function(V_a, name="Analytic Scalar")
    a_scalar = sin(pi * x[0]) * sin(pi * x[1])
    u_a.interpolate(a_scalar)

    q_a = Function(U_a, name="Analytic Flux")
    a_flux = -grad(a_scalar)
    q_a.project(a_flux)

    return a_scalar, a_flux, u_a, q_a


def solver_LDGH(mesh, degree, tau_order, a_scalar, a_flux, plot_sol=False):

    if tau_order is None or tau_order not in ("1", "1/h", "h"):
        raise ValueError("Must specify tau to be of order '1', '1/h', or 'h'")

    assert degree > 0, "Provide a degree >= 1"

    # Set up problem domain
    x = SpatialCoordinate(mesh)
    n = FacetNormal(mesh)

    # Set up function spaces
    U = VectorFunctionSpace(mesh, "DG", degree)
    V = FunctionSpace(mesh, "DG", degree)
    T = FunctionSpace(mesh, "HDiv Trace", degree)

    # Mixed space and test/trial functions
    W = U * V * T
    s = Function(W, name="solutions").assign(0.0)
    q, u, uhat = split(s)
    v, w, mu = TestFunctions(W)

    Vh = FunctionSpace(mesh, "DG", degree + 3)
    f = Function(Vh).interpolate(-div(grad(a_scalar)))

    # Determine stability parameter tau
    if tau_order == "1":
        tau = Constant(1e0)

    elif tau_order == "1/h":
        tau = 1 / FacetArea(mesh)

    elif tau_order == "h":
        tau = FacetArea(mesh)

    else:
        raise ValueError("Invalid choice of tau")

    # Numerical flux
    qhat = q + tau * (u - uhat) * n

    # Formulate the LDG-H method in UFL
    # a = (
    #     (dot(v, q) - div(v) * u) * dx
    #     + uhat("+") * jump(v, n=n) * dS
    #     + uhat * dot(v, n) * ds
    #     - dot(grad(w), q) * dx
    #     + jump(qhat, n=n) * w("+") * dS
    #     + dot(qhat, n) * w * ds
    #     # Transmission condition
    #     + mu("+") * jump(qhat, n=n) * dS
    # )
    # Model parameters
    k = Constant(1.0)
    mu = Constant(1.0)
    rho = Constant(0.0)
    g = Constant((0.0, 0.0))
    # Formulate the LS-H method in UFL
    a = (
            inner((mu / k) * q, v + (k / mu) * grad(w)) * dx
            + uhat("+") * jump(v, n=n) * dS
            + uhat * dot(v, n) * ds
            - dot(div(v), u) * dx
            + dot(grad(u), (k / mu) * grad(w)) * dx
            + inner(div(q), div(v)) * dx
            + inner(curl((mu / k) * q), curl((mu / k) * v)) * dx
            # Transmission condition
            + mu("+") * jump(qhat, n=n) * dS
    )

    L = div(v) * f * dx
    F = a - L
    hybrid_params = {
        "mat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        "pc_python_type": "firedrake.HybridizationPC",
        "hybridization": {"ksp_type": "preonly", "pc_type": "lu"},
    }
    PETSc.Sys.Print(
        "*******************************************\nSolving using static condensation.\n"
    )
    params = {
        "snes_type": "ksponly",
        "mat_type": "matfree",
        "pmat_type": "matfree",
        "ksp_type": "preonly",
        "pc_type": "python",
        # Use the static condensation PC for hybridized problems
        # and use a direct solve on the reduced system for u_hat
        "pc_python_type": "scpc.HybridSCPC",
        "hybrid_sc": {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_package": "mumps",
        },
    }

    bcs = DirichletBC(W.sub(2), Constant(0.0), "on_boundary")
    problem = NonlinearVariationalProblem(F, s, bcs=bcs)
    solver = NonlinearVariationalSolver(problem, solver_parameters=params)
    # solver = NonlinearVariationalSolver(problem, solver_parameters=hybrid_params)
    solver.solve()
    PETSc.Sys.Print("Solver finished.\n")

    # Computed flux, scalar, and trace
    q_h, u_h, uhat_h = s.split()
    u_h.rename("Scalar", "label")
    q_h.rename("Flux", "label")

    # Here we plot the solution if it is demanded
    if plot_sol == True:
        import matplotlib.pyplot as plt

        plot(q_h)
        plot(u_h)
        plt.show()

    return u_h, q_h, uhat_h


def compute_error(computed_sol, analytical_sol, var_name, norm_type="L2"):

    # Now we compute the various metrics. First we
    # simply compute the L2 error between the analytic
    # solutions and the computed ones.
    error = errornorm(analytical_sol, computed_sol, norm_type=norm_type)

    # We keep track of all metrics using a Python dictionary
    error_dictionary = {var_name: error}

    return error_dictionary


def scalar_post_processing(mesh, degree, u_h, q_h):

    # Scalar post-processing:
    # This gives an approximation in DG(k+1) via solving for
    # the solution of the local Neumann data problem:
    #
    # (grad(u), grad(w))*dx = -(q_h, grad(w))*dx
    # m(u) = m(u_h) for all elements K, where
    #
    # m(v) := measure(K)^-1 * int_K v dx.

    # NOTE: It is currently not possible to correctly formulate this
    # in UFL. However, we can introduce a Lagrange multiplier and
    # transform the local problem above into a local mixed system:
    #
    # find (u, psi) in DG(k+1) * DG(0) such that:
    #
    # (grad(u), grad(w))*dx + (psi, grad(w))*dx = -(q_h, grad(w))*dx
    # (u, phi)*dx = (u_h, phi)*dx,
    #
    # for all w, phi in DG(k+1) * DG(0).
    DGk1 = FunctionSpace(mesh, "DG", degree + 1)
    DG0 = FunctionSpace(mesh, "DG", 0)
    Wpp = DGk1 * DG0

    up, psi = TrialFunctions(Wpp)
    wp, phi = TestFunctions(Wpp)

    # Create mixed tensors:
    K = Tensor((inner(grad(up), grad(wp)) + inner(psi, wp) + inner(up, phi)) * dx)
    F = Tensor((-inner(q_h, grad(wp)) + inner(u_h, phi)) * dx)

    E = K.inv * F

    PETSc.Sys.Print("Local post-processing of the scalar variable.\n")
    u_pp = Function(DGk1, name="Post-processed scalar")
    assemble(E.blocks[0], tensor=u_pp)

    return u_pp


def flux_post_processing(mesh, degree, tau_order, u_h, q_h, uhat_h):

    # Post processing of the flux:
    # This is a modification of the local Raviart-Thomas projector.
    # We solve the local problem: find 'q_pp' in RT(k+1)(K) such that
    #
    # (q_pp, v)*dx = (q_h, v)*dx,
    # (q_pp.n, gamma)*dS = (qhat.n, gamma)*dS
    #
    # for all v, gamma in DG(k-1) * DG(k)|_{trace}. The post-processed
    # solution q_pp converges at the same rate as q_h, but is HDiv
    # conforming. For all LDG-H methods,
    # div(q_pp) converges at the rate k+1. This is a way to obtain a
    # flux with better conservation properties. For tau of order 1/h,
    # div(q_pp) converges faster than q_h.

    if tau_order is None or tau_order not in ("1", "1/h", "h"):
        raise ValueError("Must specify tau to be of order '1', '1/h', or 'h'")

    assert degree > 0, "Provide a degree >= 1"

    # Determine stability parameter tau
    if tau_order == "1":
        tau = Constant(1e0)

    elif tau_order == "1/h":
        tau = 1 / FacetArea(mesh)

    elif tau_order == "h":
        tau = FacetArea(mesh)

    else:
        raise ValueError("Invalid choice of tau")

    n = FacetNormal(mesh)
    T = FunctionSpace(mesh, "HDiv Trace", degree)

    qhat_h = q_h + tau * (u_h - uhat_h) * n
    local_RT = FiniteElement("RT", triangle, degree + 1)
    RTd = FunctionSpace(mesh, BrokenElement(local_RT))
    DGkn1 = VectorFunctionSpace(mesh, "DG", degree - 1)

    # Use the trace space already defined
    Npp = DGkn1 * T
    n_p = TrialFunction(RTd)
    vp, mu = TestFunctions(Npp)

    # Assemble the local system and invert using Slate
    A = Tensor(inner(n_p, vp) * dx + jump(n_p, n=n) * mu * dS + dot(n_p, n) * mu * ds)
    B = Tensor(inner(q_h, vp) * dx + jump(qhat_h, n=n) * mu * dS + dot(qhat_h, n) * mu * ds)

    PETSc.Sys.Print("Local post-processing of the flux.\n")
    q_pp = assemble(A.inv * B)
    q_pp.rename("Post-processed flux", "label")

    # And check the error in our new flux
    flux_pp_error = errornorm(a_flux, q_pp, norm_type="L2")

    # To verify our new flux is HDiv conforming, we also
    # evaluate its jump over mesh interiors. This should be
    # approximately zero if everything worked correctly.
    flux_pp_jump = assemble(jump(q_pp, n=n) * dS)
    PETSc.Sys.Print("Post-processing finished.\n")

    return q_pp, flux_pp_jump


def pvd_writer(var_to_write, tau_order, nelx, nely, prefix="LDGH"):
    if tau_order == "1/h":
        o = "hneg1"
    else:
        o = tau_order

    output = File("%s_tauO%s_deg%d_%dx%d.pvd" % (prefix, o, degree, nelx, nely))
    output.write(var_to_write)

    return


def writer_all_outputs(vars_to_write, tau_order, nelx, nely):
    for i in range(len(vars_to_write)):
        pvd_writer(vars_to_write[i], tau_order, nelx, nely, prefix=("var%d" % i))
    return


def writer_outputs(u_h, q_h, u_pp, q_pp, u_a, q_a, tau_order, nelx, nely, prefix="LDGH"):
    if tau_order == "1/h":
        o = "hneg1"
    else:
        o = tau_order
    output = File("%s_tauO%s_deg%d_%dx%d.pvd" % (prefix, o, degree, nelx, nely))
    output.write(u_h, q_h, u_pp, q_pp, u_a, q_a)
    return


tau_order = "1"
for deg in range(1, 4):
    scalar_errors = np.array([])
    flux_errors = np.array([])
    num_cells = np.array([])
    mesh_size = np.array([])
    degree = deg
    for i in range(2, 8):
        nel_x, nel_y = 2.0 ** i, 2.0 ** i
        mesh = UnitSquareMesh(nel_x, nel_y, quadrilateral=False)
        num_cells = np.append(num_cells, mesh.num_cells())
        mesh_size = np.append(mesh_size, nel_x)

        a_scalar, a_flux, u_a, q_a = analytical_solution(mesh, degree)
        u_h, q_h, uhat_h = solver_LDGH(mesh, degree, tau_order, a_scalar, a_flux, plot_sol=False)
        error_dictionary = {}
        error_dictionary.update(compute_error(u_h, a_scalar, "scalar_error"))
        scalar_errors = np.append(scalar_errors, error_dictionary["scalar_error"])
        error_dictionary.update(compute_error(q_h, a_flux, "flux_error"))
        flux_errors = np.append(flux_errors, error_dictionary["flux_error"])
    scalar_errors_log2 = np.log10(scalar_errors)
    flux_errors_log2 = np.log10(flux_errors)
    num_cells_log2 = np.log10(num_cells)
    mesh_size_log2 = np.log10(mesh_size)
    from scipy.stats import linregress

    scalar_slope, intercept, r_value, p_value, stderr = linregress(
        mesh_size_log2, scalar_errors_log2
    )
    print(
        "\n--------------------------------------\nDegree %d: Scalar slope error %f"
        % (degree, np.abs(scalar_slope))
    )
    flux_slope, intercept, r_value, p_value, stderr = linregress(mesh_size_log2, flux_errors_log2)
    print(
        "Degree %d: Flux slope error %f\n--------------------------------------\n"
        % (degree, np.abs(flux_slope))
    )
    plt.loglog(
        mesh_size,
        scalar_errors,
        "-o",
        label=(r"k = %d; slope = %f" % (degree, np.abs(scalar_slope))),
    )
    np.savetxt(("errors_degree%d.dat" % deg), np.transpose([num_cells, scalar_errors]))

plt.grid(True)
plt.xlabel(r"$\log (h)$")
plt.ylabel(r"$\log\left(||p-p_h||_{L^2(\Omega)}\right)$")
plt.legend(loc="best")
plt.show()
