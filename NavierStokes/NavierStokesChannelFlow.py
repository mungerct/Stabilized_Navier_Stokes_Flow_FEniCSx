#!/usr/bin/env python

'''
This python file is a stabilized Stokes flow solver that can be used to predict the output shape of 
low Reynolds number exetrusion flow. The files that are needed are the "image2gmsh3D.py" and
"image2inlet.py", which are in the "StokesFlow" folder in github. This code is made using FEniCSx
version 0.0.9, and dolfinX version 0.9.0.0 and solves stabilized Stokes flow.
The Grad-Div stabilization method is used to allow Taylor Hood (P2-P1) and lower order (P1-P1) elements 
can be used becuase of the stabilization parameters. To improve efficiency of the 
solver, the inlet boundary conditions are fully devolped flow which are generated in the image2inlet.py
file, and gmsh is used to mesh the domain.

Caleb Munger
August 2024
'''

import sys
import os
from dolfinx import mesh
from dolfinx import fem, la
from dolfinx.io import gmshio
from mpi4py import MPI
from basix.ufl import element, mixed_element
from dolfinx.fem import functionspace, dirichletbc, Function
from petsc4py import PETSc
import ufl
from ufl import div, dx, grad, inner, dot, sqrt, conditional, nabla_grad, le, sym
from image2gmsh3D import *
from image2gmsh3D import main as meshgen
from image2inlet import solve_inlet_profiles
import time
from dolfinx import log
from dolfinx.la import create_petsc_vector
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)

class NonlinearPDE_SNESProblem:
    # Directly create the Petsc nonlinear solver
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

snes_ksp_type = 'tfqmr'

# ------ Inputs ------
if len(sys.argv) == 4: # Reynolds number (use Re<10 for best results)
    Re = int(sys.argv[1])
    img_fname = sys.argv[2] # Input a .png file that is black and white (example on github) to be used as the inlet profile
    flowrate_ratio = float(sys.argv[3]) # Ratio of the two flow rates must be between 0 and 1, 0.5 is equal flow rate for both
    channel_mesh_size = 0.1
elif len(sys.argv) == 5:
    Re = int(sys.argv[1])
    img_fname = sys.argv[2]
    flowrate_ratio = float(sys.argv[3])
    channel_mesh_size = float(sys.argv[4]) # optional third argument for the element size of the 3D mesh

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

FolderName = f'NSChannelFlow_RE{Re}_MeshLC{channel_mesh_size}'
if rank == 0:
    if not os.path.exists(FolderName):
        os.makedirs(FolderName)

MPI.COMM_WORLD.Barrier()
comm.Barrier()

if rank == 0:
    print("Accepted Inputs", flush=True)

# ------ Create mesh and inlet velocity profiles ------
tstart = time.time()
uh_1, msh_1, uh_2, msh_2 = solve_inlet_profiles(img_fname, flowrate_ratio)
end = time.perf_counter()

msh = meshgen(img_fname, channel_mesh_size)
msh, _, ft = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
# msh, _, ft = gmshio.read_from_msh("ChannelMesh.msh", MPI.COMM_WORLD, 0, gdim=3)
ft.name = "Facet markers"
if rank == 0:
    print(f"Made Mesh = {time.time() - tstart: 0.2f} sec", flush=True)

# ------ Create the different finite element spaces ------
P2 = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,)) # Velocity elements for P1-P1
# P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,)) # Velocity elmeents for Taylor-Hood (P2-P1)
P1 = element("Lagrange", msh.basix_cell(), 1) # Pressure elements
V, Q = functionspace(msh, P2), functionspace(msh, P1)

if rank == 0:
    print(f"Pressure Degrees of Freedom: {Q.dofmap.index_map.size_local}", flush=True)
    print(f"Velocity Degrees of Freedom: {V.dofmap.index_map.size_local}", flush=True)


# ------- Create boundary conditions -------
''' There are 5 markers that are created by gmsh, they are used set bounadry conditions
#1 inlet_1 - Inner flow at the inlet
#2 inlet_2 - Outer flow at the inlet
#3 outlet - Outlet flow location
#4 wall - all of the walls inside and on the edge of the channel
#5 fluid - fluid marker for inside the domain, NOT used in boundary conditions'''
TH = mixed_element([P2, P1]) # Create mixed element
W = functionspace(msh, TH)
# No Slip Wall Boundary Condition
W0 = W.sub(0)
Q, _ = W0.collapse()
noslip = Function(Q) # Creating a function makes a function of zeros, which is what is needed for a no-slip BC
dofs = fem.locate_dofs_topological((W0, Q), msh.topology.dim-1, ft.find(4))
bc_wall = fem.dirichletbc(noslip, dofs, W0)

# inlet 1 boundary condition (inlet 1 is the inner channel)
'''
To make the inlet boundary conditions, the solution from the 2D inlet profile is interplotaed onto the 3D domain,
this reduces the size of the domain required. Here is a link to for an exmaple of interpolation in fenicsx 0.9
link: https://github.com/FEniCS/dolfinx/blob/9ee46f0925da2930fa76ca046d047a55555dfbff/python/test/unit/fem/test_interpolation.py#L913
'''
uh_1.x.scatter_forward()
inlet_1_velocity = fem.Function(Q)

msh_cell_map = msh.topology.index_map(msh.topology.dim)
num_cells_on_proc = msh_cell_map.size_local + msh_cell_map.num_ghosts
cells = np.arange(num_cells_on_proc, dtype=np.int32)

interpolation_data = fem.create_interpolation_data(inlet_1_velocity.function_space, uh_1.function_space,
        cells, padding=1.0e-6)

inlet_1_velocity.interpolate_nonmatching(uh_1, cells, interpolation_data)

if rank == 0:
    print("Finished Interpolating uh_1", flush=True)
dofs = fem.locate_dofs_topological((W0, Q), msh.topology.dim-1, ft.find(1))
bc_inlet_1 = dirichletbc(inlet_1_velocity, dofs, W0)

# interpolate inlet 2 boundary condition
if rank == 0:
    print("Starting to Interpolate uh_2", flush=True)
uh_2.x.scatter_forward()
inlet_2_velocity = fem.Function(Q)

msh_cell_map = msh.topology.index_map(msh.topology.dim)
num_cells_on_proc = msh_cell_map.size_local + msh_cell_map.num_ghosts
cells = np.arange(num_cells_on_proc, dtype=np.int32)

interpolation_data = fem.create_interpolation_data(inlet_2_velocity.function_space, uh_2.function_space,
        cells, padding=1.0e-6)

inlet_2_velocity.interpolate_nonmatching(uh_2, cells, interpolation_data)

if rank == 0:
    print("Finished Interpolating uh_2", flush=True)

# Inlet 2 Velocity Boundary Condition (inlet 2 is the outer channel)
dofs = fem.locate_dofs_topological((W0, Q), msh.topology.dim-1, ft.find(2))
bc_inlet_2 = dirichletbc(inlet_2_velocity, dofs, W0)

W0 = W.sub(1)
Q, _ = W0.collapse()
# Outlet Pressure Condition
dofs = fem.locate_dofs_topological((W0), msh.topology.dim-1, ft.find(3))
bc_outlet = dirichletbc(PETSc.ScalarType(0), dofs, W0)

if rank == 0:
    print("Start to Combine Boundary Conditions", flush=True)
bcs = [bc_wall, bc_inlet_1, bc_inlet_2, bc_outlet]
end = time.perf_counter()
elapsed = end - tstart

if rank == 0:
    print(f"Finihsed Combining Boundary Conditions", flush=True)


# ------ Create/Define weak form ------
dx = ufl.dx(metadata={'quadrature_degree':1}) # Reduce to 1 gauss point to increase speed (no loss of accuracy for linear elements)
W0 = W.sub(0)
Q, _ = W0.collapse()
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = Function(Q)

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

'''
For more infromation about stabilizing stokes flow, the papers "Grad-Div Stabilization for Stokes Equations" by Maxim A. Olshanskii
and "On the parameter of choice in grad-div stabilization for the stokes equations" by Eleanir W. Jenkins are recommended
'''

# ------ Assemble LHS matrix and RHS vector and solve-------
from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs = bcs, petsc_options={'ksp_type': 'bcgs', 'ksp_rtol':1e-10, 'ksp_atol':1e-10})
log.set_log_level(log.LogLevel.INFO)
U = Function(W)
U = problem.solve() # Solve the problem
log.set_log_level(log.LogLevel.WARNING)


# ------ Split the mixed solution and collapse ------
u, p = U.sub(0).collapse(), U.sub(1).collapse()
if rank == 0:
    print("Solved Stokes Flow", flush=True)

# ------ Create/Define weak form of Navier-Stokes Equations ------
del u, v, p, q, a, L, problem, f, mu_T # Clear Stokes flow variables to prevent errors

nu = 1/Re

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
# SUPG
u_norm = sqrt(dot(u,u))
tau_SUNG1 = h/(2*u_norm)
inv_tau_SUNG1 = conditional((le(u_norm, 1e-4)), 0, 1/(tau_SUNG1**r))
tau_SUNG3 = h*h/(4*nu)
inv_tau_SUNG3 = conditional((le(h*h, 1e-4)), 0, 1/(tau_SUNG3**r))

tau_SUPG = (inv_tau_SUNG1 + inv_tau_SUNG3)**(-1/r)

# LSIC
Re_UGN = u_norm*h/(2*nu)
z = conditional((le(Re_UGN, 3)), Re_UGN/3, 1)
tau_LSIC = h/2*u_norm*z

# PSPG
del u_norm
u_norm = 1
tau_SUNG1 = h/(2*u_norm)
tau_SUNG3 = h*h/(4*nu)
tau_PSPG = (1/(tau_SUNG1**r)+1/(tau_SUNG3**r))**(-1/r)

if rank == 0:
    print("Made Stabilzation Coefficients", flush=True)

a = inner(dot(u, nabla_grad(u)),v) * dx # Advection
a += nu*inner(grad(u),grad(v)) * dx # Diffusion
a -= inner(p,div(v)) * dx # Pressure
a += inner(q,div(u)) * dx # Incompressibility
res = dot(u, nabla_grad(u)) - nu*div(sym(grad(u))) + grad(p) # Momentum residual
a += tau_SUPG*inner(dot(u, nabla_grad(v)), res) * dx # SUPG
a += tau_SUPG*inner(grad(q), res) * dx # PSPG
a += tau_LSIC*inner(div(v), div(u)) * dx # LSIC

if rank == 0:
    print("Made Weak Form", flush=True)

dw = ufl.TrialFunction(W)
dF = ufl.derivative(a, w, dw)

w.interpolate(U) # Interpolate the Stokes flow solution to set as intial condition for Navier-Stokes flow
if rank == 0:
    print("Interpolated Stokes Flow", flush=True)

problem = NonlinearPDE_SNESProblem(a, w, bcs)

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

if comm.rank == 0:
    print('Running SNES solver')

comm.barrier()

t_start = time.time()

snes.solve(None, w.x.petsc_vec)

t_stop = time.time()

if comm.rank == 0:
    print(f'Num SNES iterations: {snes.getIterationNumber()}')
    print(f'SNES termination reason: {snes.getConvergedReason()}')

snes.destroy()
b.destroy()
J.destroy()

log.set_log_level(log.LogLevel.WARNING)
if rank == 0:
    print('Solved Navier-Stokes', flush=True)
# Split the mixed solution and collapse
u = w.sub(0).collapse() # Velocity
p = w.sub(1).collapse() # Pressure

tstop = time.time()
if rank == 0:
    print(f"run time = {time.time() - tstart: 0.2f} sec", flush=True)

MPI.COMM_WORLD.Barrier()
if rank == 0:
    print('Saving files', flush=True)

# ------ Save the solutions to both a .xdmf and .h5 file
# Save the pressure field
from dolfinx.io import XDMFFile
from basix.ufl import element as VectorElement
with XDMFFile(MPI.COMM_WORLD, f"{FolderName}/Re{Re}ChannelPressure.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    P3 = VectorElement("Lagrange", msh.basix_cell(), 1)
    u1 = Function(functionspace(msh, P3))
    u1.interpolate(p)
    u1.name = 'Pressure'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u1)

# Save the velocity field
with XDMFFile(MPI.COMM_WORLD, f"{FolderName}/Re{Re}ChannelVelocity.xdmf", "w") as pfile_xdmf:
    u.x.scatter_forward()
    P4 = VectorElement("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
    u2 = Function(functionspace(msh, P4))
    u2.interpolate(u)
    u2.name = 'Velocity'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u2)

'''
if rank == 0:
    print(f"Re={Re}\n", flush=True)
    print(f"img_filename={img_fname}\n", flush=True)
    print(f"Flowrate Ratio={flowrate_ratio}\n", flush=True)
    print(f"Channel Mesh Size={channel_mesh_size}\n", flush=True)
    print(f"Pressure DOFs: {Q.dofmap.index_map.size_local}\n", flush=True)
    print(f"Velocity DOFs: {V.dofmap.index_map.size_local}\n", flush=True)
    size = comm.Get_size()
    print(f"{size} Cores Used\n", flush=True)
    print(f"Run Time = {time.time() - tstart: 0.2f} sec\n", flush=True)
'''
# redirct stdout to a file text file in python


with open(f"{FolderName}/RunParameters.txt", "w") as file:
    file.write(f"Re={Re}\n")
    file.write(f"img_filename={img_fname}\n")
    file.write(f"Flowrate Ratio={flowrate_ratio}\n")
    file.write(f"Channel Mesh Size={channel_mesh_size}\n")
    file.write(f"Pressure DOFs: {Q.dofmap.index_map.size_local}\n")
    file.write(f"Velocity DOFs: {V.dofmap.index_map.size_local}\n")
    size = comm.Get_size()
    file.write(f"{size} Cores Used\n")
    file.write(f"Run Time = {time.time() - tstart: 0.2f} sec\n")


MPI.COMM_WORLD.Barrier()
MPI.COMM_WORLD.Abort(0)