#!/usr/bin/env python

import sys
import os
from dolfinx import fem, la
from dolfinx.io import gmshio
from mpi4py import MPI
from basix.ufl import element, mixed_element
from dolfinx.fem import functionspace, dirichletbc, Function
from petsc4py import PETSc
import ufl
from ufl import div, dx, grad, inner, dot, sqrt, conditional, nabla_grad, le, sym
import time
from dolfinx import log
import numpy as np

from dolfinx.fem import (Function, functionspace,
                         assemble_scalar, dirichletbc, form)
from dolfinx.io import gmshio
from ufl import (FacetNormal, Measure,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, sym)

gmsh_model = sys.argv[1]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

msh, _, ft = gmshio.read_from_msh(gmsh_model, MPI.COMM_WORLD, 0, gdim=2)

P2 = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,)) # Velocity elements for P1-P1
P1 = element("Lagrange", msh.basix_cell(), 1) # Pressure elements
V, Q = functionspace(msh, P2), functionspace(msh, P1)

if rank == 0:
    print(f"Pressure Degrees of Freedom: {Q.dofmap.index_map.size_local}", flush=True)
    print(f"Velocity Degrees of Freedom: {V.dofmap.index_map.size_local}", flush=True)

# Use dS instead of ds to get lift/drag over a 2D surface
# use dolfinx.io.gmshio.read_from_msh to import mesh from gmsh
# Install gmsh and make mesh in gmsh

TH = mixed_element([P2, P1]) # Create mixed element
W = functionspace(msh, TH)
# No Slip Wall Boundary Condition
W0 = W.sub(0)
V1, _ = W0.collapse() # Velocity subspace
W1 = W.sub(1)
Q1, _ = W1.collapse() # Pressure subspace

gdim = 2
fdim = 1
def InletVelocity(x):
    values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = (4 * x[1] * 0.3 * (0.41 - x[1]) / (0.41**2))
    return values

# Maker Numbers
# Inlet = 2
# Oulet = 3
# Wall = 4
# Obstacle = 5

# Inlet
u_inlet = Function(V1)
u_inlet.interpolate(InletVelocity)
bcu_inflow = dirichletbc(u_inlet, fem.locate_dofs_topological((W0, V1), fdim, ft.find(2)), W0)
inlet_dofs = fem.locate_dofs_topological(V1, fdim, ft.find(2))
# print('bcu inflow')
# print(inlet_dofs)
# Walls
u_nonslip = Function(V1) # by default zeros
bcu_walls = dirichletbc(u_nonslip, fem.locate_dofs_topological((W0, V1), fdim, ft.find(4)), W0)
wall_dofs = fem.locate_dofs_topological(V1, fdim, ft.find(4))
# print('bc walls')
# print(wall_dofs)
# Obstacle
bcu_obstacle = dirichletbc(u_nonslip, fem.locate_dofs_topological((W0, V1), fdim, ft.find(5)), W0)
bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
obstacle_dofs = fem.locate_dofs_topological(V1, fdim, ft.find(5))
# print('bc obstacle')
# print(obstacle_dofs)
# Outlet
OuletValue = Function(Q1)
bcp_outlet = dirichletbc(OuletValue, fem.locate_dofs_topological((W1, Q1), fdim, ft.find(3)), W1)
bcp = [bcp_outlet]
outlet_dofs = fem.locate_dofs_topological(Q1, fdim, ft.find(3))
# print('bc outlet')
# print(outlet_dofs)

bc = [bcu_inflow, bcu_walls, bcu_obstacle]

# ------ Create/Define weak form ------
dx = ufl.dx(metadata={'quadrature_degree':2}) # Reduce to 1 gauss point to increase speed (no loss of accuracy for linear elements)
W0_NS = W.sub(0)
V_NS, _ = W0_NS.collapse()
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = Function(V_NS)

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

# ------ Assemble LHS matrix and RHS vector and solve-------
from dolfinx.fem.petsc import LinearProblem
# problem = LinearProblem(a, L, bcs = bc, petsc_options={'ksp_type': 'bcgs', 'ksp_rtol':1e-10, 'ksp_atol':1e-10})
problem = LinearProblem(a,L,bcs=bc, petsc_options = {
    'ksp_type':'preonly',
    'pc_type':'lu',
    'pc_factor_mat_solver_type':'mumps',
    'ksp_monitor':''
})
log.set_log_level(log.LogLevel.INFO)
U = Function(W)
U = problem.solve() # Solve the problem
log.set_log_level(log.LogLevel.WARNING)

# ------ Split the mixed solution and collapse ------
u, p = U.sub(0).collapse(), U.sub(1).collapse()
if rank == 0:
    print("Solved Stokes Flow", flush=True)


# ------ Create/Define weak form ------
dx = ufl.dx(metadata={'quadrature_degree':2}) # Reduce to 2 gauss point to increase speed (no loss of accuracy for linear elements)
W0_NS = W.sub(0)
V_NS, _ = W0_NS.collapse()
w = Function(W)
(u, p) = ufl.split(w)
(v, q) = ufl.TestFunctions(W)
f = Function(V_NS)

nu = 1e-3
r = 2
h = ufl.CellDiameter(msh)
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
w.interpolate(U)

from dolfinx.fem.petsc import NonlinearProblem
problem = NonlinearProblem(a, w, bcs=bc, J=dF)
from dolfinx.nls.petsc import NewtonSolver

solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-9
solver.report = True

log.set_log_level(log.LogLevel.INFO)

ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
# opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}ksp_type"] = "tfqmr"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# Compute the solution
solver.solve(w)
log.set_log_level(log.LogLevel.WARNING)

# Split the mixed solution and collapse
u = w.sub(0).collapse() # Velocity
p = w.sub(1).collapse() # Pressure


n = -FacetNormal(msh)  # Normal pointing out of obstacle
dObs = Measure("ds", domain=msh, subdomain_data=ft, subdomain_id=5)
u_t = inner(as_vector((n[1], -n[0])), u)
drag_coeff = form(2 / (0.1*(0.2**2)) * (nu * inner(grad(u_t), n) * n[1] - p * n[0]) * dObs)
lift_coeff = form(-2 / (0.1 * (0.2**2)) * (nu * inner(grad(u_t), n) * n[0] + p * n[1]) * dObs)
Cl_error = form(((-2 / (0.1 * (0.2**2)) * (nu * inner(grad(u_t), n) * n[0] + p * n[1]) - 0.010618948146)/0.010618948146) * dObs)
Cd_error = form(((2 / (0.1*(0.2**2)) * (nu * inner(grad(u_t), n) * n[1] - p * n[0]) - 5.57953523384)/5.57953523384) * dObs)

drag_coeff = msh.comm.gather(assemble_scalar(drag_coeff), root=0)
lift_coeff = msh.comm.gather(assemble_scalar(lift_coeff), root=0)
Cl_error = msh.comm.gather(assemble_scalar(Cl_error), root=0)
Cd_error = msh.comm.gather(assemble_scalar(Cd_error), root=0)

if rank == 0:
    print(f"Coefficient of Lift: {lift_coeff}", flush=True)
    print(f"Cl Percent Error: {Cl_error}", flush=True)
    print(f"Coefficient of Drag: {drag_coeff}", flush=True)
    print(f"Cd Percent Error: {Cd_error}", flush=True)

from dolfinx.io import XDMFFile
from basix.ufl import element as VectorElement

with XDMFFile(MPI.COMM_WORLD, f"DFG2DValidationPressure.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    P3 = VectorElement("Lagrange", msh.basix_cell(), 1)
    u1 = Function(functionspace(msh, P3))
    u1.interpolate(p)
    u1.name = 'Pressure'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u1)

# Save the velocity field
with XDMFFile(MPI.COMM_WORLD, f"DFG2DValidationVelocity.xdmf", "w") as pfile_xdmf:
    u.x.scatter_forward()
    P4 = VectorElement("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
    u2 = Function(functionspace(msh, P4))
    u2.interpolate(u)
    u2.name = 'Velocity'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u2)