from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import log
from dolfinx.fem import Function, dirichletbc, functionspace, locate_dofs_topological, locate_dofs_geometrical
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner, dot, nabla_grad
from mpi4py import MPI

# Create mesh
msh = create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                       [16, 16], CellType.triangle)


# Function to mark x = 0, x = 1 and y = 0
def noslip_boundary(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
                         np.isclose(x[1], 0.0))


# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)

# Lid velocity
def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))

P2 = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,)) # Velocity elements for P1-P1
# P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,)) # Velocity elmeents for Taylor-Hood (P2-P1)
P1 = element("Lagrange", msh.basix_cell(), 1) # Pressure elements
V, Q = functionspace(msh, P2), functionspace(msh, P1)
print(f"There are this Many Degrees of Freedom in the Pressure Nodes: {Q.dofmap.index_map.size_local}")

# Create the Taylor-Hood function space
TH = mixed_element([P2, P1])
W = functionspace(msh, TH)

# No slip boundary condition
W0, _ = W.sub(0).collapse()
noslip = Function(W0)
facets = locate_entities_boundary(msh, 1, noslip_boundary)
dofs = locate_dofs_topological((W.sub(0), W0), 1, facets)
bc0 = dirichletbc(noslip, dofs, W.sub(0))

# Driving velocity condition u = (1, 0) on top boundary (y = 1)
lid_velocity = Function(W0)
lid_velocity.interpolate(lid_velocity_expression)
facets = locate_entities_boundary(msh, 1, lid)
dofs = locate_dofs_topological((W.sub(0), W0), 1, facets)
bc1 = dirichletbc(lid_velocity, dofs, W.sub(0))

zero = Function(Q)
dofs = locate_dofs_geometrical(
    (W.sub(1), Q), lambda x: np.isclose(x.T, [0, 0, 0]).all(axis=1))
bc2 = dirichletbc(zero, dofs, W.sub(1))

bcs = [bc0, bc1, bc2]

# ------ Create/Define weak form ------
W0 = W.sub(0)
Q, _ = W0.collapse()
w = Function(W)
(u, p) = ufl.split(w)
(v, q) = ufl.TestFunctions(W)
f = Function(Q)

nu = 100 # Viscocity (Reynolds number equals 1/nu)
a = inner(dot(u, nabla_grad(u)),v) * dx # Advection
a += nu*inner(grad(u),grad(v)) * dx # Diffusion
a -= inner(p,div(v)) * dx # Pressure
a -= inner(q,div(u)) * dx # Incompressibility

dw = ufl.TrialFunction(W)
dF = ufl.derivative(a, w, dw)

from dolfinx.fem.petsc import NonlinearProblem
problem = NonlinearProblem(a, w, bcs=bcs, J=dF)
from dolfinx.nls.petsc import NewtonSolver

solver = (MPI.COMM_WORLD, problem)
log.set_log_level(log.LogLevel.INFO)

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# Compute the solution
solver.solve(w)

# Split the mixed solution and collapse
u = w.sub(0).collapse()
p = w.sub(1).collapse()

'''
from dolfinx.fem.petsc import NonlinearProblem # https://fenicsproject.discourse.group/t/error-in-solving-steady-navier-stokes-equation/10224
problem = NonlinearProblem(a, L, bcs)
from dolfinx.nls.petsc import NewtonSolver

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True

log.set_log_level(log.LogLevel.INFO)

# Stabilization parameters per Andre Massing
h = ufl.CellDiameter(msh)
Beta = 0.2
mu_T = Beta*h*h # Stabilization coefficient
a = inner(grad(u), grad(v)) * dx
a -= inner(p, div(v)) * dx
a += inner(div(u), q) * dx
a += mu_T*inner(grad(p), grad(q)) * dx # Stabilization term

L = inner(f,v) * dx
L -= mu_T * inner(f, grad(q)) * dx # Stabilization  term

from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs = bcs, petsc_options={'ksp_type': 'bcgs', 'ksp_rtol':1e-10, 'ksp_atol':1e-10})
U = Function(W)
U = problem.solve() # Solve the problem
print('Solved Stokes Flow')
'''