#!/usr/bin/env python

import sys
import os
from dolfinx import mesh
from dolfinx import fem, la
from dolfinx.io import gmshio
from mpi4py import MPI
from basix.ufl import element, mixed_element
from dolfinx.fem import functionspace, dirichletbc, Function
from dolfinx.la import create_petsc_vector
from petsc4py import PETSc
import ufl
from ufl import div, dx, grad, inner, dot, sqrt, conditional, nabla_grad, le, sym, tr, inv, Jacobian
import time
from dolfinx import log
import numpy as np

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element

from dolfinx.cpp.mesh import to_type, cell_entity_type
from dolfinx.fem import (Constant, Function, functionspace,
                         assemble_scalar, dirichletbc, form, locate_dofs_topological, set_bc)
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.graph import adjacencylist
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import (VTXWriter, distribute_entity_data, gmshio)
from dolfinx.mesh import create_mesh, meshtags_from_entities
from ufl import (FacetNormal, Identity, Measure, TestFunction, TrialFunction,
                 as_vector, div, dot, ds, dx, inner, lhs, grad, nabla_grad, rhs, sym, system)

class NonlinearPDE_SNESProblem:
    def __init__(self, F, u, bc):
        V = u.function_space
        du = ufl.TrialFunction(V)
        self.L = fem.form(F)
        self.a = fem.form(ufl.derivative(F, u, du))
        self.bc = bc
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                       mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], [self.bc], [x], -1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bc, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        from dolfinx.fem.petsc import assemble_matrix

        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bc)
        J.assemble()

#gmsh_model = sys.argv[1]
gmsh_model = 'dfg_pillar_3D.msh'
snes_ksp_type = 'tfqmr'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

msh, _, ft = gmshio.read_from_msh(gmsh_model, MPI.COMM_WORLD, 0, gdim=3)

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

gdim = 3
fdim = 2
def InletVelocity(x):
    values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = (4 * x[1] * (0.41 - x[1]) / (0.41**2))*((4 * x[2] * (0.41 - x[2]) / (0.41**2))) * 0.45
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
dx = ufl.dx(metadata={'quadrature_degree':2}) # Reduce to 2 gauss point to increase speed (no loss of accuracy for linear elements)
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

problem = LinearProblem(a, L, bcs = bc, petsc_options={
    'ksp_type': 'fgmres',
    'pc_type': 'asm',
    'ksp_monitor':''
    }
)
log.set_log_level(log.LogLevel.INFO)
U = Function(W)
U = problem.solve() # Solve the problem
log.set_log_level(log.LogLevel.WARNING)

# ------ Split the mixed solution and collapse ------
u, p = U.sub(0).collapse(), U.sub(1).collapse()
if rank == 0:
    print("Solved Stokes Flow", flush=True)


# ------ Create/Define weak form ------
dx = ufl.dx(metadata={'quadrature_degree':2}) # Reduce to 1 gauss point to increase speed (no loss of accuracy for linear elements)
W0_NS = W.sub(0)
V_NS, _ = W0_NS.collapse()
w = Function(W)
(u, p) = ufl.split(w)
(v, q) = ufl.TestFunctions(W)
f = Function(V_NS)

nu = 0.001
mu = nu
'''
# SUPG and PSPG (h-based formulation)
r = 2
h = ufl.CellDiameter(msh)
u_norm = sqrt(dot(u,u))
tau_SUNG1 = h/(2*u_norm)
inv_tau_SUNG1 = conditional((le(u_norm, 1e-8)), 0, 1/(tau_SUNG1**r))
tau_SUNG3 = h*h/(4*nu)
tau_SUPG = (inv_tau_SUNG1+1/(tau_SUNG3**r))**(-1/r)

# LSIC
Re_UGN = u_norm*h/(2*nu)
z = conditional((le(Re_UGN, 3)), Re_UGN/3, 1)
tau_LSIC = h/2*u_norm*z
'''

a = inner(dot(u, nabla_grad(u)),v) * dx # Advection
a += nu*inner(grad(u),grad(v)) * dx # Diffusion
a -= inner(p,div(v)) * dx # Pressure
a += inner(q,div(u)) * dx # Incompressibility

'''
a += tau_SUPG*inner(dot(u, nabla_grad(v)), res) * dx # SUPG
a += tau_SUPG*inner(grad(q), res) * dx # PSPG
a += tau_LSIC*inner(div(v), div(u)) * dx # LSIC
'''

# G-based SUPG/PSPG/LSIC (element metric tensor)
x = ufl.SpatialCoordinate(msh)
dxi_dy = inv(Jacobian(msh))
dxi_dx = dxi_dy * inv(grad(x))
G = (dxi_dx.T)*dxi_dx

# SUPG + PSPG
Ci = 36.0
tau_SUPS = 1.0 / sqrt(inner(u, G*u) + Ci*(mu**2)*inner(G, G))

sigma = 2.*mu*ufl.sym(grad(u)) - p*ufl.Identity(len(u))
res_M = dot(u, grad(u)) - div(sigma)
#res_M = dot(u, nabla_grad(u)) - nu*div(sym(grad(u))) + grad(p) # Momentum residual

a += inner(tau_SUPS * res_M, dot(u, grad(v)) + grad(q)) * dx

# LSIC
v_LSIC = 1.0 / (tr(G) * tau_SUPS)

res_C = div(u)
a += v_LSIC * div(v) * res_C * dx

# Set Stokes flow solution as initial guess
w.interpolate(U)

u_IC = w.sub(0).collapse() # Velocity
p_IC = w.sub(1).collapse() # Pressure

from dolfinx.io import XDMFFile
from basix.ufl import element as VectorElement

'''
with XDMFFile(MPI.COMM_WORLD, f"DFGValidationPressureIntialGuess.xdmf", "w") as pfile_xdmf:
    p_IC.x.scatter_forward()
    p_IC.name = 'Pressure'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(p_IC)

# Save the velocity field
with XDMFFile(MPI.COMM_WORLD, f"DFGValidationVelocityIntialGuess.xdmf", "w") as pfile_xdmf:
    u_IC.x.scatter_forward()
    u_IC.name = 'Velocity'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u_IC)

'''
problem = NonlinearPDE_SNESProblem(a, w, bc)

b = create_petsc_vector(W.dofmap.index_map, W.dofmap.index_map_bs)
J = create_matrix(problem.a)

snes = PETSc.SNES().create()
opts = PETSc.Options()
opts["snes_monitor"] = None
snes.setFromOptions()
snes.setFunction(problem.F, b)
snes.setJacobian(problem.J, J)

snes.setTolerances(rtol=1.0e-8, atol=1e-8, max_it=30)
snes.getKSP().setType(snes_ksp_type)
snes.getKSP().setTolerances(rtol=1.0e-8)

#if pc == 'hypre':
#    snes.getKSP().getPC().setType(PETSc.PC.Type.HYPRE)
#    snes.getKSP().getPC().setHYPREType("boomeramg")
#elif pc == 'asm':
#    snes.getKSP().getPC().setType("asm")
#    snes.getKSP().getPC().setASMType(PETSc.PC.ASMType.BASIC)

if comm.rank == 0:
    print('Running SNES solver')

comm.barrier()

t_start = time.time()

snes.solve(None, w.x.petsc_vec)

t_stop = time.time()

#assert snes.getConvergedReason() > 0
#assert snes.getIterationNumber() < 6
if comm.rank == 0:
    print(f'Num SNES iterations: {snes.getIterationNumber()}')
    print(f'SNES termination reason: {snes.getConvergedReason()}')

snes.destroy()
b.destroy()
J.destroy()

'''
dw = ufl.TrialFunction(W)
dF = ufl.derivative(a, w, dw)

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
#opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}ksp_type"] = "tfqmr"
#opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# Compute the solution
solver.solve(w)
log.set_log_level(log.LogLevel.WARNING)
'''

# Split the mixed solution and collapse
u = w.sub(0).collapse() # Velocity
p = w.sub(1).collapse() # Pressure

Lc = 0.1*0.41
Uc = 0.2

rho = 1
mu = nu
n = -ufl.FacetNormal(msh)
stress = -p * ufl.Identity(3) + 2.0 * mu * ufl.sym(ufl.grad(u))
traction = ufl.dot(stress, n)

drag_expr = traction[0]
lift_expr = traction[1]

dObs = ufl.Measure("ds", domain=msh, subdomain_data=ft, subdomain_id=5)
drag_form = form(drag_expr * dObs)
lift_form = form(lift_expr * dObs)

F_drag = fem.assemble_scalar(drag_form)
F_lift = fem.assemble_scalar(lift_form)

F_drag = comm.allreduce(F_drag, op=MPI.SUM)
F_lift = comm.allreduce(F_lift, op=MPI.SUM)

drag_coeff = 2* F_drag / (rho * Uc**2 * Lc)
lift_coeff = 2* F_lift / (rho * Uc**2 * Lc)

'''
n = -FacetNormal(msh)  # Normal pointing out of obstacle
dObs = Measure("ds", domain=msh, subdomain_data=ft.find(5))
# dObs = Measure("ds", ft.find(5), domain=msh)
u_t = inner(as_vector((n[2], -n[1], n[0])), u)
drag = form(2 / 0.1 * (1 * inner(grad(u_t), n) * n[1] - p * n[0]) * dObs)
lift = form(-2 / 0.1 * (1 * inner(grad(u_t), n) * n[0] + p * n[1]) * dObs)
'''

if rank == 0:
    print(f"Coefficient of Lift: {lift_coeff}", flush=True)
    print(f"Coefficient of Drag: {drag_coeff}", flush=True)

from dolfinx.io import XDMFFile
from basix.ufl import element as VectorElement


with XDMFFile(MPI.COMM_WORLD, f"DFGValidationPressureNavierStokes.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    p.name = 'Pressure'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(p)

# Save the velocity field
with XDMFFile(MPI.COMM_WORLD, f"DFGValidationVelocityNavierStokes.xdmf", "w") as pfile_xdmf:
    u.x.scatter_forward()
    u.name = 'Velocity'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u)