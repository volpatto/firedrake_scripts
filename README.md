# Firedrake Scripts

This repo contains several MWE (minimal working examples) scripts which exemplify the usage of Finite Element Methods (FEM)
to solve some problems I faced around or the ones I found interesting. The implementation
is done with [Firedrake Project](https://www.firedrakeproject.org/) framework, a very
powerful FEM toolkit that provides a Python high-level syntax (more specifically, a Domain
Specific Language - DSL) and automatic code generation.

# Available scripts

* 1D cases:
    * Poisson with two variations, one shows how to check Stiffness Matrix inputs;
    * Linear parabolic (transient Poisson);
    * Transient linear reaction-diffusion with a simple exponential growth as reaction 
    term (inspired by simplified Tumor Growth);
    * Gray-Scott non-linear transient reaction-diffusion system of PDE (two coupled PDEs),
    a very interesting case which exhibits self-replication in autocatalytic gel reactors.
    The code reproduces studies provided in this [paper](https://www.sciencedirect.com/science/article/pii/S0022247X15007957?via%3Dihub);
    * Linear Darcy flow with homogeneous permeability coefficient;
    * Linear Darcy flow with heterogeneous (discontinuous) permeability coefficient;
    * Non-linear transient Darcy flow coupled with Peng-Robinson Equation of States (PR-EoS) to mimic
    compressible flow in rigid porous medium with low permeability (inspired in Shale rocks).
* 2D cases:
    * Non-linear steady-state Helmholtz based on the example provided by 
    [Firedrake's repo](https://github.com/firedrakeproject/firedrake);
    * Non-linear Darcy flow inspired by compressible gas flow in a rigid porous medium
    with low permeability, such as reservoir composed by Shale rocks. There are
    slightly distinct sub-cases: 
        - With .pvd (Paraview compatible) output files (solved with Newton method);
        - With matplotlib/Firedrake output (solved with Newton method);
        - A case exemplifying how to implement Picard linearization within Firedrake for
        non-linear problems;
    * (Primal) Linear elasticity problem based on the example provided by [Firedrake's
    notebooks](https://nbviewer.jupyter.org/github/firedrakeproject/firedrake/blob/master/docs/notebooks/example-elasticity.ipynb).

## About Me

* Name: Diego Volpatto;
* Email: [volpatto@lncc.br](volpatto@lncc.br).

I'm a Numerical Developer at [ESSS](https://github.com/ESSS). Also I currently pursue
a DSc in Computational Modeling at [LNCC](http://www.lncc.br/). Check it out if you please!