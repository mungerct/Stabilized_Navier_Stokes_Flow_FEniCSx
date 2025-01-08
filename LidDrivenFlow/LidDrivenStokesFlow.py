from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, la
from dolfinx.fem import (Constant, Function, dirichletbc,
                         extract_function_spaces, form, functionspace,
                         locate_dofs_topological)
from dolfinx.fem.petsc import assemble_matrix_block, assemble_vector_block
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import div, dx, grad, inner

# Create mesh
msh = create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([1, 1])],
                       [64, 64], CellType.triangle)


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

# P2 = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,)) # Velocity elements for P1-P1
P2 = element("Lagrange", msh.basix_cell(), 2, shape=(msh.geometry.dim,)) # Velocity elmeents for Taylor-Hood (P2-P1)
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

bcs = [bc0, bc1]

# ------ Create/Define weak form ------
W0 = W.sub(0)
Q, _ = W0.collapse()
(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
f = Function(Q)

# Stabilization parameters per Andre Massing
nu = 0.01
h = ufl.CellDiameter(msh)
a0 = 1/3
mu_T = a0*h*h/(4*nu) # Stabilization coefficient
a = nu*inner(grad(u), grad(v)) * dx
a -= inner(p, div(v)) * dx
a += inner(div(u), q) * dx
a += mu_T*inner(grad(p), grad(q)) * dx # GLS stabilization term

L = inner(f,v) * dx
L -= mu_T * inner(f, grad(q)) * dx # GLS stabilization  term

from dolfinx.fem.petsc import LinearProblem
problem = LinearProblem(a, L, bcs = bcs, petsc_options={'ksp_type': 'bcgs', 'ksp_rtol':1e-10, 'ksp_atol':1e-10})
U = Function(W)
U = problem.solve() # Solve the problem

# ------ Split the mixed solution and collapse ------
u, p = U.sub(0).collapse(), U.sub(1).collapse()
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
with XDMFFile(MPI.COMM_WORLD, "StokesLidDrivenPressureHighRe.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    P3 = VectorElement("Lagrange", msh.basix_cell(), 1)
    u1 = Function(functionspace(msh, P3))
    u1.interpolate(p)
    u1.name = 'Pressure'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u1)

# Save the velocity field
with XDMFFile(MPI.COMM_WORLD, "StokesLidDrivenVelocityHighRe.xdmf", "w") as pfile_xdmf:
    u.x.scatter_forward()
    P4 = VectorElement("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
    u2 = Function(functionspace(msh, P4))
    u2.interpolate(u)
    u2.name = 'Velocity'
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(u2)