#!/usr/bin/env python
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import log 
from dolfinx.fem import Function, dirichletbc, functionspace, locate_dofs_topological, locate_dofs_geometrical
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner, dot, nabla_grad, sym, sqrt, conditional, le
from mpi4py import MPI
import time
import sys


if len(sys.argv) == 3:
    Re = int(sys.argv[1])
    NumCells = int(sys.argv[2])

elif len(sys.argv) == 2:
    Re = int(sys.argv[1])
    NumCells = 64

if MPI.COMM_WORLD.rank == 0:
    tstart = time.time()

# Create mesh
msh = create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                    [NumCells, NumCells], CellType.triangle)

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
V, Q = functionspace(msh, P2), functionspace(msh, P1) # V is velocity space, Q is pressure space
if MPI.COMM_WORLD.rank == 0:
    print(f"Pressure Degress of Freedom: {Q.dofmap.index_map.size_local}")
    print(f"Velocity Degress of Freedom: {V.dofmap.index_map.size_local}")

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

# Create 0 pressure boundary condition at 0,0
zero = Function(Q)
dofs = locate_dofs_geometrical(
    (W.sub(1), Q), lambda x: np.isclose(x.T, [0, 0, 0]).all(axis=1))
bc2 = dirichletbc(zero, dofs, W.sub(1))

bcs = [bc0, bc1, bc2]

# Create stokes flow solution
W0_Stokes = W.sub(0)
V_Stokes, _ = W0_Stokes.collapse()
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = Function(V_Stokes)

# Stabilization parameters per Andre Massing
nu = 1/Re # Viscocity (Reynolds number equals 1/nu)
h = ufl.CellDiameter(msh)
a0 = 1/3
mu_T = a0*h*h/(4*nu) # Stabilization coefficient

dx = ufl.dx(metadata={'quadrature_degree':2})
a = nu*inner(grad(u), grad(v)) * dx
a -= inner(p, div(v)) * dx
a += inner(div(u), q) * dx
a += mu_T*inner(grad(p), grad(q)) * dx # Stabilization term

L = inner(f, v) * dx
L -= mu_T * inner(f, grad(q)) * dx # Stabilization  term

from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs = bcs, petsc_options={'ksp_type': 'bcgs', 'ksp_rtol':1e-10, 'ksp_atol':1e-10})
U = Function(W)
log.set_log_level(log.LogLevel.INFO)
U = problem.solve() # Solve the problem
log.set_log_level(log.LogLevel.WARNING)
print('Solved Stokes Flow')

# ------ Create/Define weak form of Navier-Stokes Equations ------
del u, v, p, q, a, L, problem, f, mu_T # Clear Stokes flow variables to prevent errors

W0_NS = W.sub(0)
V_NS, _ = W0_NS.collapse()
w = Function(W)
(u, p) = ufl.split(w)
(v, q) = ufl.TestFunctions(W)
f = Function(V_NS)

# Stabiliztion parameter from "CALCULATION OF THE STABILIZATION PARAMETERS IN FINITE ELEMENT FORMULATIONS OF FLOW PROBLEMS"
# By Tayfun E. Tezduyar, used the UGN based parameters found in section 7 of his paper, for more information about finite elemnet
# stabilization, see the youtube playlist by Dr. Stein Stoter, https://www.youtube.com/playlist?list=PLMHTjE57oyvpkTPG8ON1w6BChBoapsZTA

r = 2
# SUPG and PSPG
u_norm = sqrt(dot(u,u))
tau_SUNG1 = h/(2*u_norm)
inv_tau_SUNG1 = conditional((le(u_norm, 1e-8)), 0, 1/(tau_SUNG1**r))
tau_SUNG3 = h*h/(4*nu)
tau_SUPG = (inv_tau_SUNG1+1/(tau_SUNG3**r))**(-1/r)

# LSIC
Re_UGN = u_norm*h/(2*nu)
z = conditional((le(Re_UGN, 3)), Re_UGN/3, 1)
tau_LSIC = h/2*u_norm*z


a = inner(dot(u, nabla_grad(u)),v) * dx # Advection
a += nu*inner(grad(u),grad(v)) * dx # Diffusion
a -= inner(p,div(v)) * dx # Pressure
a += inner(q,div(u)) * dx # Incompressibility
res = dot(u, nabla_grad(u)) - nu*div(sym(grad(u))) + grad(p) # Momentum residual
a += tau_SUPG*inner(dot(u, nabla_grad(v)), res) * dx # SUPG
a += tau_SUPG*inner(grad(q), res) * dx # PSPG
a += tau_LSIC*inner(div(v), div(u)) * dx # LSIC

dw = ufl.TrialFunction(W)
dF = ufl.derivative(a, w, dw)

w.interpolate(U) # Interpolate the Stokes flow solution to set as intial condition for Navier-Stokes flow

from dolfinx.fem.petsc import NonlinearProblem
problem = NonlinearProblem(a, w, bcs=bcs, J=dF)
from dolfinx.nls.petsc import NewtonSolver

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-9
solver.report = True

log.set_log_level(log.LogLevel.INFO)

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# Compute the solution
solver.solve(w)
log.set_log_level(log.LogLevel.WARNING)

# Split the mixed solution and collapse
u = w.sub(0).collapse() # Velocity
p = w.sub(1).collapse() # Pressure

tstop = time.time()
if MPI.COMM_WORLD.rank == 0:
    print(f"run time = {tstop - tstart: 0.2f} sec")

# ------ Save the solutions to both a .xdmf and .h5 file
# Save the pressure field
from dolfinx.io import XDMFFile
from basix.ufl import element as VectorElement

with XDMFFile(MPI.COMM_WORLD, f"NavierStokesLidDrivenPressureLinear{Re}.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    P3 = VectorElement("Lagrange", msh.basix_cell(), 1)
    u1 = Function(functionspace(msh, P3))
    u1.interpolate(p)
    u1.name = 'Pressure'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u1)

# Save the velocity field
with XDMFFile(MPI.COMM_WORLD, f"NavierStokesLidDrivenPressureVelocity{Re}.xdmf", "w") as pfile_xdmf:
    u.x.scatter_forward()
    P4 = VectorElement("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
    u2 = Function(functionspace(msh, P4))
    u2.interpolate(u)
    u2.name = 'Velocity'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u2)