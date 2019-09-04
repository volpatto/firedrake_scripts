# -*- coding: utf-8 -*-
"""
Este código soluciona, por meio do SUPG, o problema de Advecção-Difusão estacionário 
com fronteiras de Dirichlet homogêneas e fonte unitária:

    -k*u''(x) + b*u'(x) = 1
     u(x_left) = 0
     u(x_right) = 0

Utiliza-se valores que geram elevado número de Péclet: k = 10^-8 e b = 1.
Obs.: Os parâmetros de estabilização implementados só são específicos de elementos
lineares. Para graus superiores, os valores devem ser revisados.

Autor: Diego T. Volpatto
Email: volpatto@lncc.br
"""
# Bibliotecas importadas
from firedrake import *
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

# Definindo a malha no intervalo
x_left = 0.0
x_right = 1.0
numel = 100
mesh = IntervalMesh(numel, x_left, x_right)  # IntervalMesh(num_de_elementos, inf_intervalo, sup_intervalo)
mesh_ref = IntervalMesh(100, x_left, x_right)

# Definindo o espaco das funcoes de Galerkin Continuo de grau = p
p = 1
V = FunctionSpace(mesh, "CG", p)
Vref = FunctionSpace(mesh_ref, "CG", p)

# Valor a ser prescrito nas fronteiras (Dirichlet)
u1, u2 = 0.0, 0.0
g_left = Constant(u1)
g_right = Constant(u2)

# Marcando a fronteira em right com g no espaco V
bc_left = DirichletBC(V, g_left, 1)
bc_right = DirichletBC(V, g_right, 2)
dirichlet_condition = [bc_left, bc_right]

# Declarando o termo de fonte
f = Constant(1)
#x = SpatialCoordinate(mesh)
#f = Function(V)
#f.project(x[0])


# Setando os espaco das funcoes candidatas a solucao e funcoes teste
u = TrialFunction(V)
v = TestFunction(V)

# Parametros do problema
k = Constant(1e-8)
b = Constant(1.0)

# Estabelecendo a forma bilinear
a = k*inner(grad(v), grad(u))*dx + b*grad(u)[0]*v*dx

# Declarando a forma linear
L = f*v*dx

# *** Adicionando os termos de estabilização do SUPG ***
# Parâmetros de Estabilização (baseado em Franca et al. 1992)
m_k = 1.0/3.0
h_k = CellDiameter(mesh) # Parametro do tamanho da malha
b_norm = abs(b)
Pe_k = m_k*b_norm*h_k/(2.0*k)
one = Constant(1.0)
eps_k = conditional(gt(Pe_k, one), one, Pe_k)
tau_k = h_k/(2.0*b_norm) * eps_k
# Termos de Estabilização adicionais
a += inner((b*grad(u)[0] - k*div(grad(u))), tau_k*b*grad(v)[0])*dx
# Uma forma alternativa mais simples válida para o caso 1D:
#a += (b*grad(u)[0] - k*div(grad(u)))*tau_k*b*grad(v)[0]*dx
L += f*tau_k*b*grad(v)[0]*dx

# Discretizando o problema variacional
u_sol = Function(V)
problem = LinearVariationalProblem(a, L, u_sol, dirichlet_condition)  # LinearVariationalProblem(lhs, rhs, incognita, cond_cont_essencial)

# Solucionando o problema
solver = LinearVariationalSolver(problem)
solver.solve()

# *** Visualizando com o Matplotlib ***
plot(u_sol, marker='x', label='Approx')
#plot(u_e, label='Exact')
# Configuracoes de fontes no grafico
#plt.rc('text',usetex=True)
plt.rc('font', size=14)
# Plotando
plt.xlim(x_left, x_right)   # Limites do eixo x
#plt.ylim(np.min(u_sol.vector().get_local()), np.max(u_e.vector().get_local()))  # Limites do eixo y
plt.grid(True, linestyle='--')  # Ativa o grid do grafico
plt.xlabel(r'$x$')  # Legenda do eixo x
plt.ylabel(r'$u(x)$')   # Legenda do eixo y
#plt.legend(loc='best',borderpad=0.5)    # Ativa legenda no grafico e diz para se posicionar na melhor localizacao detectada
plt.show()  # Exibe o grafico em tela
