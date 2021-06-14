from firedrake import *
from petsc4py import PETSc

OptDB = PETSc.Options()
OptDB['draw_pause'] = -1  # to open X-window without immediately close it

n = 1
mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "P", 1)
u = TrialFunction(V)
v = TestFunction(V)
a = u*v*dx
A = assemble(a)
Ap = A.petscmat
viewer = PETSc.Viewer(Ap.getType())
viewer.createDraw()

draw = PETSc.Viewer.DRAW(Ap.comm)
draw(Ap)

# Print on file
viewer.createASCII(name='test_viewer.txt')
viewer.pushFormat(viewer.Format.ASCII_DENSE)
Ap.view(viewer=viewer)

"""
Mat Object: 1 MPI processes
  type: seqaij
row 0: (0, 0.166667)  (1, 0.0416667)  (2, 0.0833333)  (3, 0.0416667)
row 1: (0, 0.0416667)  (1, 0.0833333)  (2, 0.0416667)
row 2: (0, 0.0833333)  (1, 0.0416667)  (2, 0.166667)  (3, 0.0416667)
row 3: (0, 0.0416667)  (2, 0.0416667)  (3, 0.0833333)
"""

iset = PETSc.IS().createGeneral([0, 1])  # select indicies 0 and 1
iset2 = PETSc.IS().createGeneral([1, 0])  # select indicies 1 and 0
Ap1, = Ap.createSubMatrices(iset, iscols=iset)

# Plot on screen (approach 1)
viewer.createDraw()
Ap1.view(viewer=viewer)

# Plot on the screen (approach 2)
#draw = PETSc.Viewer.DRAW(Ap1.comm)
#draw(Ap1)

"""
Mat Object: 1 MPI processes
  type: seqaij
row 0: (0, 0.166667)  (1, 0.0416667)
row 1: (0, 0.0416667)  (1, 0.0833333)
"""

Ap2, = Ap.createSubMatrices(iset2, iscols=iset2)

viewer.createDraw()
Ap2.view(viewer=viewer)

# Print on file
viewer2 = PETSc.Viewer(Ap2.getType())
viewer2.createASCII(name='test_viewer2.csv')
viewer2.pushFormat(viewer2.Format.ASCII_DENSE)
Ap2.view(viewer=viewer2)

"""
Mat Object: 1 MPI processes
  type: seqaij
row 0: (0, 0.0833333)  (1, 0.0416667) 
row 1: (0, 0.0416667)  (1, 0.166667) 
"""

