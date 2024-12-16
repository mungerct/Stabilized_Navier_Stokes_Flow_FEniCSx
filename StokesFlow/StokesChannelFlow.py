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
from ufl import div, dx, grad, inner
from image2gmsh3D import *
from image2gmsh3D import main as meshgen
from image2inlet import solve_inlet_profiles
import time

# ------ Inputs ------
if len(sys.argv) == 3:
    img_fname = sys.argv[1] # Input a .png file that is black and white (example on github) to be used as the inlet profile
    flowrate_ratio = float(sys.argv[2]) # Ratio of the two flow rates must be between 0 and 1, 0.5 is equal flow rate for both
    channel_mesh_size = 0.25
elif len(sys.argv) == 4:
    img_fname = sys.argv[1]
    flowrate_ratio = float(sys.argv[2])
    channel_mesh_size = float(sys.argv[3]) # optional third argument for the element size of the 3D mesh

print("Accepted Inputs, Begining Solving Inlet Profiles")

# ------ Create mesh and inlet velocity profiles ------
start = time.perf_counter()
uh_1, msh_1, uh_2, msh_2 = solve_inlet_profiles(img_fname, flowrate_ratio)
end = time.perf_counter()
elapsed = end - start
print(f"Inlet Profile Solved, Starting to Make Mesh. {elapsed:.6f} Seconds to Solve Inlet Profiles")
start = time.perf_counter()
msh = meshgen(img_fname, channel_mesh_size)
msh, _, ft = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
# msh, _, ft = gmshio.read_from_msh("ChannelMesh.msh", MPI.COMM_WORLD, 0, gdim=3)
ft.name = "Facet markers"
end = time.perf_counter()
elapsed = end - start
print(f"Finished Making Mesh. {elapsed:.6f} Seconds to Make Mesh and Turn it Into FEniCSx Object")
start = time.perf_counter()

# ------ Create the different finite element spaces ------
P2 = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,)) # Velocity elements for P1-P1
# P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,)) # Velocity elmeents for Taylor-Hood (P2-P1)
P1 = element("Lagrange", msh.basix_cell(), 1) # Pressure elements
V, Q = functionspace(msh, P2), functionspace(msh, P1)
print(f"There are this Many Degrees of Freedom in the Pressure Nodes: {Q.dofmap.index_map.size_local}")


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

print("Finished Interpolating uh_1")
dofs = fem.locate_dofs_topological((W0, Q), msh.topology.dim-1, ft.find(1))
bc_inlet_1 = dirichletbc(inlet_1_velocity, dofs, W0)

# interpolate inlet 2 boundary condition
print("\nStarting to Interpolate uh_2")
uh_2.x.scatter_forward()
inlet_2_velocity = fem.Function(Q)

msh_cell_map = msh.topology.index_map(msh.topology.dim)
num_cells_on_proc = msh_cell_map.size_local + msh_cell_map.num_ghosts
cells = np.arange(num_cells_on_proc, dtype=np.int32)

interpolation_data = fem.create_interpolation_data(inlet_2_velocity.function_space, uh_2.function_space,
        cells, padding=1.0e-6)

inlet_2_velocity.interpolate_nonmatching(uh_2, cells, interpolation_data)
print("Finished Interpolating uh_2")

# Inlet 2 Velocity Boundary Condition (inlet 2 is the outer channel)
dofs = fem.locate_dofs_topological((W0, Q), msh.topology.dim-1, ft.find(2))
bc_inlet_2 = dirichletbc(inlet_2_velocity, dofs, W0)

W0 = W.sub(1)
Q, _ = W0.collapse()
# Outlet Pressure Condition
dofs = fem.locate_dofs_topological((W0), msh.topology.dim-1, ft.find(3))
bc_outlet = dirichletbc(PETSc.ScalarType(0), dofs, W0)
print("\nStart to Combine Boundary Conditions")
bcs = [bc_wall, bc_inlet_1, bc_inlet_2, bc_outlet]
end = time.perf_counter()
elapsed = end - start
print(f"Finihsed Combining Boundary Conditions. {elapsed:.6f} Seconds to Apply all Boundary Conditions")


# ------ Create/Define weak form ------
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


start = time.perf_counter()
# ------ Assemble LHS matrix and RHS vector and solve-------
print("\nStart Assembling Stiffness Matrix and Forcing Vector")
from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs = bcs, petsc_options={'ksp_type': 'bcgs', 'ksp_rtol':1e-10, 'ksp_atol':1e-10})
U = Function(W)
U = problem.solve() # Solve the problem



# ------ Split the mixed solution and collapse ------
u, p = U.sub(0).collapse(), U.sub(1).collapse()
end = time.perf_counter()
elapsed = end - start
print(f'{elapsed:.6f} Seconds to Solve The System')
# Compute norms
norm_u, norm_p = la.norm(u.x), la.norm(p.x)
norm_inf_u, inf_norm_p = la.norm(u.x, type=la.Norm.linf), la.norm(p.x, type=la.Norm.linf)
if MPI.COMM_WORLD.rank == 0:
    print(f"\nL2 Norm of velocity coefficient vector: {norm_u}")
    print(f"Infinite Norm of velocity coefficient vector: {norm_inf_u}")
    print(f"L2 Norm of pressure coefficient vector: {norm_p}")
    print(f"Infinite Norm of pressure coefficient vector: {inf_norm_p}")

print("\nFinished Solving, Saving Solution Field")



# ------ Save the solutions to both a .xdmf and .h5 file
# Save the pressure field
from dolfinx.io import XDMFFile
from basix.ufl import element as VectorElement
with XDMFFile(MPI.COMM_WORLD, "StokesChannelPressure.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    P3 = VectorElement("Lagrange", msh.basix_cell(), 1)
    u1 = Function(functionspace(msh, P3))
    u1.interpolate(p)
    u1.name = 'Pressure'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u1)

# Save the velocity field
with XDMFFile(MPI.COMM_WORLD, "StokesChannelVelocity.xdmf", "w") as pfile_xdmf:
    u.x.scatter_forward()
    P4 = VectorElement("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
    u2 = Function(functionspace(msh, P4))
    u2.interpolate(u)
    u2.name = 'Velocity'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u2)