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
from ufl import div, dx, grad, inner, dot, sqrt, conditional, nabla_grad, le, sym, tr, inv, Jacobian
from image2gmsh3D import *
from image2gmsh3D import main as meshgen
from image2inlet import solve_inlet_profiles
import time
from dolfinx import log
from dolfinx.la import create_petsc_vector
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_vector, create_matrix, set_bc)
from dolfinx.io import XDMFFile
from basix.ufl import element as VectorElement

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
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def parse_arguments():
    if len(sys.argv) not in [4, 5]:
        raise ValueError("Usage: script.py <Re> <img_fname> <flowrate_ratio> [<channel_mesh_size>]")

    Re = int(sys.argv[1])
    img_fname = sys.argv[2]
    flowrate_ratio = float(sys.argv[3])
    channel_mesh_size = float(sys.argv[4]) if len(sys.argv) == 5 else 0.1

    return Re, img_fname, flowrate_ratio, channel_mesh_size

def create_output_directory(folder_name, rank):
    if rank == 0 and not os.path.exists(folder_name):
        os.makedirs(folder_name)
    MPI.COMM_WORLD.Barrier()
    if rank == 0:
        print("Accepted Inputs", flush=True)

def generate_inlet_profiles(img_fname, flowrate_ratio):
    uh_1, msh_1, uh_2, msh_2 = solve_inlet_profiles(img_fname, flowrate_ratio)
    return uh_1, msh_1, uh_2, msh_2


def generate_mesh(img_fname, channel_mesh_size):
    if rank == 0:
        print('Meshing', flush = True)
    msh = meshgen(img_fname, channel_mesh_size)
    msh, _, ft = gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    ft.name = "Facet markers"
    if rank == 0:
        num_elem = msh.topology.index_map(msh.topology.dim).size_global
        print(f'Num elem: {num_elem}', flush = True)
    return msh, ft


def define_function_spaces(msh):
    P2 = element("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
    P1 = element("Lagrange", msh.basix_cell(), 1)
    V = functionspace(msh, P2)
    Q = functionspace(msh, P1)
    return V, Q


def create_boundary_conditions(msh, ft, V, Q, uh_1, uh_2):
    TH = mixed_element([V.ufl_element(), Q.ufl_element()])
    W = functionspace(msh, TH)
    W0 = W.sub(0)
    V_interp, _ = W0.collapse()

    noslip = Function(V_interp)
    bc_wall = dirichletbc(noslip, fem.locate_dofs_topological((W0, V_interp), msh.topology.dim - 1, ft.find(4)), W0)

    inlet_1_velocity = interpolate_inlet_to_3d(uh_1, V_interp, msh)
    bc_inlet_1 = dirichletbc(inlet_1_velocity, fem.locate_dofs_topological((W0, V_interp), msh.topology.dim - 1, ft.find(1)), W0)

    inlet_2_velocity = interpolate_inlet_to_3d(uh_2, V_interp, msh)
    bc_inlet_2 = dirichletbc(inlet_2_velocity, fem.locate_dofs_topological((W0, V_interp), msh.topology.dim - 1, ft.find(2)), W0)

    W1 = W.sub(1)
    Q_interp, _ = W1.collapse()
    bc_outlet = dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(W1, msh.topology.dim - 1, ft.find(3)), W1)

    bcs = [bc_wall, bc_inlet_1, bc_inlet_2, bc_outlet]
    return W, bcs


def interpolate_inlet_to_3d(uh, V, msh):
    uh.x.scatter_forward()
    v_interp = fem.Function(V)
    msh_cell_map = msh.topology.index_map(msh.topology.dim)
    cells = np.arange(msh_cell_map.size_local + msh_cell_map.num_ghosts, dtype=np.int32)
    interp_data = fem.create_interpolation_data(V, uh.function_space, cells, padding=1e-6)
    v_interp.interpolate_nonmatching(uh, cells, interp_data)
    return v_interp


def setup_stokes_weak_form(W, msh):
    dx = ufl.dx(metadata={'quadrature_degree': 2})
    W0 = W.sub(0)
    V, _ = W0.collapse()
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    f = Function(V)

    h = ufl.CellDiameter(msh)
    mu_T = 0.2 * h * h
    a = inner(grad(u), grad(v)) * dx - inner(p, div(v)) * dx + inner(div(u), q) * dx + mu_T * inner(grad(p), grad(q)) * dx
    L = inner(f, v) * dx - mu_T * inner(f, grad(q)) * dx
    return a, L, V


def interpolate_initial_guess(U, Uold, V, Q, msh):
    uold, pold = Uold.sub(0).collapse(), Uold.sub(1).collapse()
    uold.x.scatter_forward()
    pold.x.scatter_forward()

    velocity_interp = fem.Function(V)
    pressure_interp = fem.Function(Q)

    msh_cell_map = msh.topology.index_map(msh.topology.dim)
    cells = np.arange(msh_cell_map.size_local + msh_cell_map.num_ghosts, dtype=np.int32)

    interp_data_v = fem.create_interpolation_data(V, uold.function_space, cells, padding=1e-6)
    interp_data_p = fem.create_interpolation_data(Q, pold.function_space, cells, padding=1e-6)

    velocity_interp.interpolate_nonmatching(uold, cells, interp_data_v)
    pressure_interp.interpolate_nonmatching(pold, cells, interp_data_p)

    U.sub(0).interpolate(velocity_interp)
    U.sub(1).interpolate(pressure_interp)
    return U


def solve_stokes_problem(a, L, bcs, W, Uold=None, msh=None):
    problem = fem.petsc.LinearProblem(a, L, bcs=bcs, petsc_options={
        'ksp_type': 'tfqmr',
        'pc_type': 'asm',
        'ksp_monitor': ''
    })

    log.set_log_level(log.LogLevel.INFO)
    U = Function(W)

    if Uold:
        V, Q = W.sub(0).collapse()[0], W.sub(1).collapse()[0]
        U = interpolate_initial_guess(U, Uold, V, Q, msh)

    if rank == 0:
        print("Starting Linear Solve", flush=True)

    U = problem.solve()
    log.set_log_level(log.LogLevel.WARNING)
    if rank == 0:
        print("Finished Linear Solve", flush=True)
    return U

def define_navier_stokes_form(W, msh, Re, U_stokes=None, U_coarse=None):
    dx = ufl.dx(metadata={'quadrature_degree': 2})
    nu = 1 / Re
    V_NS, _ = W.sub(0).collapse()

    w = Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    f = Function(V_NS)

    # Metric tensor for stabilization
    x = ufl.SpatialCoordinate(msh)
    dxi_dy = inv(Jacobian(msh))
    dxi_dx = dxi_dy * inv(grad(x))
    G = (dxi_dx.T) * dxi_dx

    Ci = 36.0
    tau_SUPS = 1.0 / sqrt(inner(u, G * u) + Ci * (nu ** 2) * inner(G, G))

    sigma = 2 * nu * ufl.sym(grad(u)) - p * ufl.Identity(len(u))
    res_M = dot(u, grad(u)) - div(sigma)

    a = inner(dot(u, nabla_grad(u)), v) * dx
    a += nu * inner(grad(u), grad(v)) * dx
    a -= inner(p, div(v)) * dx
    a += inner(q, div(u)) * dx
    a += inner(tau_SUPS * res_M, dot(u, grad(v)) + grad(q)) * dx

    v_LSIC = 1.0 / (tr(G) * tau_SUPS)
    res_C = div(u)
    a += v_LSIC * div(v) * res_C * dx

    dw = ufl.TrialFunction(W)
    dF = ufl.derivative(a, w, dw)
    if U_stokes:
        if rank == 0:
            print("Interpolating Stokes Flow", flush=True)
        w.interpolate(U_stokes)

    if U_coarse:
        if rank == 0:
            print("Interpolating Coarse NS Flow", flush=True)
        V, Q = W.sub(0).collapse()[0], W.sub(1).collapse()[0]
        w = interpolate_initial_guess(w, U_coarse, V, Q, msh)
    
    return a, w, dF

def solve_navier_stokes(a, w, dF, bcs, W, snes_ksp_type, comm, rank):
    problem = NonlinearPDE_SNESProblem(a, w, bcs)

    b = create_petsc_vector(W.dofmap.index_map, W.dofmap.index_map_bs)
    J = create_matrix(problem.a)

    snes = PETSc.SNES().create()
    opts = PETSc.Options()
    opts["snes_monitor"] = None
    snes.setFromOptions()
    snes.setFunction(problem.F, b)
    snes.setJacobian(problem.J, J)

    snes.setTolerances(rtol=1e-8, atol=1e-8, max_it=30)
    snes.getKSP().setType(snes_ksp_type)
    snes.getKSP().setTolerances(rtol=1e-8)

    if rank == 0:
        print("Running SNES solver", flush=True)

    comm.barrier()
    t_start = time.time()
    if rank == 0:
        print("Start Nonlinear Solve", flush=True)

    snes.solve(None, w.x.petsc_vec)

    t_stop = time.time()
    if rank == 0:
        print(f"Num SNES iterations: {snes.getIterationNumber()}", flush=True)
        print(f"SNES termination reason: {snes.getConvergedReason()}", flush=True)
        print(f"Navier-Stokes solve time: {t_stop - t_start:.2f} sec", flush=True)

    snes.destroy()
    b.destroy()
    J.destroy()

    if rank == 0:
        print("Finished Nonlinear Solve", flush=True)

    log.set_log_level(log.LogLevel.WARNING)

    u = w.sub(0).collapse()
    p = w.sub(1).collapse()
    return w, u, p

def save_navier_stokes_solution(u, p, msh, FolderName, Re):

    u.x.scatter_forward()
    p.x.scatter_forward()
    if rank == 0:
        print('Writing solution', flush=True)

    with XDMFFile(MPI.COMM_WORLD, f"{FolderName}/Re{Re}ChannelPressure.xdmf", "w") as pfile_xdmf:
        P3 = VectorElement("Lagrange", msh.basix_cell(), 1)
        p_out = Function(functionspace(msh, P3))
        p_out.interpolate(p)
        p_out.name = "Pressure"
        pfile_xdmf.write_mesh(msh)
        pfile_xdmf.write_function(p_out)

    with XDMFFile(MPI.COMM_WORLD, f"{FolderName}/Re{Re}ChannelVelocity.xdmf", "w") as ufile_xdmf:
        P4 = VectorElement("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
        u_out = Function(functionspace(msh, P4))
        u_out.interpolate(u)
        u_out.name = "Velocity"
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(u_out)

def write_run_metadata(FolderName, Re, img_fname, flowrate_ratio, channel_mesh_size, V, Q):
    with open(f"{FolderName}/RunParameters.txt", "w") as file:
        file.write(f"Re={Re}\n")
        file.write(f"img_filename={img_fname}\n")
        file.write(f"Flowrate Ratio={flowrate_ratio}\n")
        file.write(f"Channel Mesh Size={channel_mesh_size}\n")
        file.write(f"Pressure DOFs: {Q.dofmap.index_map.size_local}\n")
        file.write(f"Velocity DOFs: {V.dofmap.index_map.size_local}\n")
        file.write(f"{comm.Get_size()} Cores Used\n")

def main():
    # Get Inputs
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    Re, img_fname, flowrate_ratio, channel_mesh_size = parse_arguments()
    folder_name = f'NSChannelFlow_RE{Re}_MeshLC{channel_mesh_size}'
    create_output_directory(folder_name, rank)
    
    # Solve Stokes Flow
    uh_1, msh_1, uh_2, msh_2 = generate_inlet_profiles(img_fname, flowrate_ratio)
    msh, ft = generate_mesh(img_fname, 0.1)
    V, Q = define_function_spaces(msh)
    W, bcs = create_boundary_conditions(msh, ft, V, Q, uh_1, uh_2)
    a, L, V = setup_stokes_weak_form(W, msh)
    U_stokes = solve_stokes_problem(a, L, bcs, W)

    # Solve Coarse Navier Stokes
    a, w, dF = define_navier_stokes_form(W, msh, Re, U_stokes = U_stokes)
    w_coarse, u, p = solve_navier_stokes(a, w, dF, bcs, W, snes_ksp_type, comm, rank)

    # Solve Navier Stokes With User Defined Mesh
    msh, ft = generate_mesh(img_fname, channel_mesh_size)
    V, Q = define_function_spaces(msh)
    W, bcs = create_boundary_conditions(msh, ft, V, Q, uh_1, uh_2)
    a, w, dF = define_navier_stokes_form(W, msh, Re, U_coarse = w_coarse)
    w, u, p = solve_navier_stokes(a, w, dF, bcs, W, snes_ksp_type, comm, rank)

    save_navier_stokes_solution(u, p, msh, folder_name, Re)
    write_run_metadata(folder_name, Re, img_fname, flowrate_ratio, channel_mesh_size, V, Q)

    return w, W, msh

if __name__ == "__main__":
    main()