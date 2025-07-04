# Firedrake Scripts (NOTE: this repo is not being tested following `firedrake` updates)

This repo contains several MWE (minimal working examples) scripts which exemplify the usage of Finite Element Methods (FEM)
to solve some problems I faced around or the ones I found interesting. The implementation
is done with [Firedrake Project](https://www.firedrakeproject.org/) framework, a very
powerful FEM toolkit that provides a Python high-level syntax (more specifically, a Domain
Specific Language - DSL) and automatic code generation.

## Available scripts (outdated)

* **1D cases**:
    * Poisson with two variations, one shows how to check the Stiffness Matrix inputs;
    * Linear parabolic (transient Poisson);
    * A simple linear transport problem (advection + diffusion) stabilized with SUPG; 
    * Transient linear reaction-diffusion with a simple exponential growth as reaction 
    term (inspired by simplified Tumor Growth). Available time stepping examples:
        - Crank-Nicolson;
        - 2nd order Backward Differentiation (BDF2).
    * Gray-Scott non-linear transient reaction-diffusion system of PDE (two coupled PDEs),
    a very interesting case which exhibits self-replication in autocatalytic gel reactors.
    The code reproduces studies provided in this [paper](https://www.sciencedirect.com/science/article/pii/S0022247X15007957?via%3Dihub).
    Currently I provide two different implementations for time stepping integration method:
        - Implicit Euler;
        - Crank-Nicolson (shows overall better performance and accuracy than Implicit Euler);
        - SDIRK4, a singly diagonally implicit Runge-Kutta scheme of order 4 (see [here](https://ntrs.nasa.gov/search.jsp?R=20160005923)).
        This method is suitable for stiff problems. As Gray-Scott is highly non-linear in reaction terms, it may
        present some stiff behavior. The method performs better than Crank-Nicolson, but demands more execution time.
    * Linear Darcy flow with homogeneous permeability coefficient;
    * Linear Darcy flow with heterogeneous (discontinuous) permeability coefficient;
    * Non-linear transient Darcy flow coupled with Peng-Robinson Equation of States (PR-EoS) to mimic
    compressible flow in rigid porous medium with low permeability (inspired by Shale rock reservoirs);
    * A generic reaction-diffusion with high Damkohler number (low diffusion coefficient,
    high reaction rate coefficient) stabilized with GGLS (Galerkin-gradient/least-squares)
    which was proposed [in this paper](https://www.sciencedirect.com/science/article/pii/0045782589900856);
    * A simplified Population Dynamics case (reaction-diffusion, advection can be considered) with Allee effect 
    reproducing the results available [in this paper](https://www.sciencedirect.com/science/article/pii/S0304380005003741)
    and exact solution from [this one](https://www.sciencedirect.com/science/article/pii/S0025556403000981).
    The study is performed only in reaction-diffusion case, although modification to address
    advective contribution is straightforward (and already present in the code). To solve
    the problem, the classical continuous Galerkin method is used. To integrate over time,
    there are three MWE provided:
        - Crank-Nicolson;
        - 2nd order Backward Differentiation (BFD2);
        - SDIRK4, a singly diagonally implicit Runge-Kutta scheme of order 4. This method can handle stiff problems,
        which are required to solve problems with reaction and complex dynamics. The implementation is based on 
        [this paper](https://www.sciencedirect.com/science/article/pii/S0021999107000861). A general review of DIRK 
        family methods can be found in [this report from NASA](https://ntrs.nasa.gov/search.jsp?R=20160005923).
* **2D cases**:
    * Non-linear steady-state Helmholtz based on the example provided by 
    [Firedrake's repo](https://github.com/firedrakeproject/firedrake);
    * A simple linear transport problem (advection + diffusion) stabilized with SUPG;
    * Non-linear Darcy flow inspired by compressible gas flow in a rigid porous medium
    with low permeability, such as a reservoir composed of Shale rocks. There are
    slightly distinct sub-cases: 
        - With .pvd (Paraview compatible) output files (solved with Newton method);
        - With matplotlib/Firedrake output (solved with Newton method);
        - A case exemplifying how to implement Picard linearization within Firedrake for
        non-linear problems;
    * (Primal) Linear elasticity problem based on the example provided by [Firedrake's
    notebooks](https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/example-elasticity.ipynb);
    * The extensions to 2D of a generic scalar reaction-diffusion with a high Damkohler number
    stabilized with the [GGLS](https://www.sciencedirect.com/science/article/pii/0045782589900856)
    method;
    * Stabilized mixed Finite Element method for Darcy flow as proposed by [Masud and Hughes](https://www.sciencedirect.com/science/article/pii/S0045782502003717);
    * Stabilized mixed discontinuous Galerkin method for Darcy flow as suggested by [Hughes et al](https://www.sciencedirect.com/science/article/pii/S0045782505002732);
    * An unconditionally stable mixed finite element method for Darcy flow named as CGLS, which was proposed by
    Correa and Loula in [this paper](https://www.sciencedirect.com/science/article/pii/S0045782507004768).

## About Me

* Name: Diego Volpatto;
* Email: [volpatto@lncc.br](volpatto@lncc.br).

I'm a Staff Research Scientist at the Brazilian National Laboratory for Scientific Computing ([LNCC]([https://github.com/ESSS](https://www.gov.br/lncc/pt-br))). 
Feel free to contact me.
