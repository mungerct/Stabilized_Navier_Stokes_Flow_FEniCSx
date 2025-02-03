#!/usr/bin/env python

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

gmsh_model = sys.argv[1]

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
V, _ = W0.collapse() # Velocity subspace
W1 = W.sub(1)
Q, _ = W1.collapse() # Pressure subspace

gdim = 3
fdim = 2
def InletVelocity(x):
    values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
    values[0] = 4 * x[1] * (0.41 - x[1]) / (0.41**2) * 0.45
    return values

# Maker Numbers
# Inlet = 2
# Oulet = 3
# Wall = 4
# Obstacle = 5

# Inlet
u_inlet = Function(V)
u_inlet.interpolate(InletVelocity)
bcu_inflow = dirichletbc(u_inlet, fem.locate_dofs_topological(V, fdim, ft.find(2)))
# Walls
u_nonslip = np.array((0,) * msh.geometry.dim, dtype=PETSc.ScalarType)
bcu_walls = dirichletbc(u_nonslip, fem.locate_dofs_topological(V, fdim, ft.find(4)), V)
# Obstacle
bcu_obstacle = dirichletbc(u_nonslip, fem.locate_dofs_topological(V, fdim, ft.find(5)), V)
bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
# Outlet
bcp_outlet = dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(Q, fdim, ft.find(3)), Q)
bcp = [bcp_outlet]

bc = [bcu_inflow, bcu_walls, bcu_obstacle, bcp_outlet]
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

# ------ Assemble LHS matrix and RHS vector and solve-------
from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs = bc, petsc_options={'ksp_type': 'bcgs', 'ksp_rtol':1e-10, 'ksp_atol':1e-10})
log.set_log_level(log.LogLevel.INFO)
U = Function(W)
U = problem.solve() # Solve the problem
log.set_log_level(log.LogLevel.WARNING)

# ------ Split the mixed solution and collapse ------
u, p = U.sub(0).collapse(), U.sub(1).collapse()
if rank == 0:
    print("Solved Stokes Flow", flush=True)