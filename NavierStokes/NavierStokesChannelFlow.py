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
from PIL import Image

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
    img_fname = img_fname.removeprefix(".")
    current_dir = os.getcwd()
    img_fname = current_dir + img_fname
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
    # Create the weak form of the Naiver-Stokes equation, a GLS stabilization method is used to help find a solution
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

    Ci = 36.0 # Stabilization constant, value is used in literature, it isn't exact and mirror adjustments will still yield results
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
    
    return a, w, dF, V_NS

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
        print("Running SNES solver", flush = True)

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

from dolfinx.io import XDMFFile

def save_navier_stokes_solution(u, p, msh, FolderName, Re, comm, rank):
    if rank == 0:
        print(f"[Rank {rank}] Starting save_navier_stokes_solution()", flush=True)

    u.x.scatter_forward()
    p.x.scatter_forward()

    # Interpolate on all ranks (parallel)
    P3 = VectorElement("Lagrange", msh.basix_cell(), 1)
    p_out = Function(functionspace(msh, P3))
    p_out.interpolate(p)

    P4 = VectorElement("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
    u_out = Function(functionspace(msh, P4))
    u_out.interpolate(u)

    # Open XDMF files in parallel (using comm, not MPI.COMM_SELF)
    with XDMFFile(comm, f"{FolderName}/Re{Re}ChannelPressure.xdmf", "w") as pfile_xdmf:
        p_out.name = "Pressure"
        pfile_xdmf.write_mesh(msh)
        pfile_xdmf.write_function(p_out)

    with XDMFFile(comm, f"{FolderName}/Re{Re}ChannelVelocity.xdmf", "w") as ufile_xdmf:
        u_out.name = "Velocity"
        ufile_xdmf.write_mesh(msh)
        ufile_xdmf.write_function(u_out)
    if rank == 0:
        print(f"[Rank {rank}] Solution writing complete.", flush=True)

    # Barrier to synchronize all ranks before proceeding
    comm.Barrier()

'''
def save_navier_stokes_solution(u, p, msh, FolderName, Re, comm, rank):
    print(f"[Rank {rank}] Starting save_navier_stokes_solution()", flush=True)

    u.x.scatter_forward()
    p.x.scatter_forward()

    # Interpolate in parallel on all ranks
    P3 = VectorElement("Lagrange", msh.basix_cell(), 1)
    p_out = Function(functionspace(msh, P3))
    p_out.interpolate(p)

    P4 = VectorElement("Lagrange", msh.basix_cell(), 1, shape=(msh.geometry.dim,))
    u_out = Function(functionspace(msh, P4))
    u_out.interpolate(u)
    if rank == 0:
        with XDMFFile(MPI.COMM_SELF, f"{FolderName}/Re{Re}ChannelPressure.xdmf", "w") as pfile_xdmf:
            p_out.name = "Pressure"
            pfile_xdmf.write_mesh(msh)
            pfile_xdmf.write_function(p_out)

        with XDMFFile(MPI.COMM_SELF, f"{FolderName}/Re{Re}ChannelVelocity.xdmf", "w") as ufile_xdmf:
            u_out.name = "Velocity"
            ufile_xdmf.write_mesh(msh)
            ufile_xdmf.write_function(u_out)

        print(f"[Rank {rank}] Solution writing complete.", flush=True)
    else:
        print(f"[Rank {rank}] Skipping file write", flush=True)

    # Optional: Add a barrier to sync all ranks here if you want to make sure all finish before continuing
    print(f"[Rank {rank}] Before final MPI Barrier", flush=True)
    comm.Barrier()
    print(f"[Rank {rank}] After final MPI Barrier", flush=True)
'''

def write_run_metadata(FolderName, Re, img_fname, flowrate_ratio, channel_mesh_size, V, Q, img_name, comm, rank):
    if rank == 0:
        import os
        from PIL import Image

        try:
            # Write metadata file
            filepath = os.path.join(FolderName, "RunParameters.txt")
            with open(filepath, "w") as file:
                file.write(f"Re={Re}\n")
                file.write(f"img_filename={img_fname}\n")
                file.write(f"Flowrate Ratio={flowrate_ratio}\n")
                file.write(f"Channel Mesh Size={channel_mesh_size}\n")
                file.write(f"Pressure DOFs: {Q.dofmap.index_map.size_local}\n")
                file.write(f"Velocity DOFs: {V.dofmap.index_map.size_local}\n")
                file.write(f"{comm.Get_size()} Cores Used\n")

            # Save image copy in output folder
            img = Image.open(img_fname)
            save_path = os.path.join(FolderName, f"{img_name}.png")
            img.save(save_path, format="PNG")

            print(f"[Rank {rank}] Run metadata and image saved to {FolderName}")
        except Exception as e:
            print(f"[Rank {rank}] ERROR in write_run_metadata: {e}")
            raise
    else:
        # Other ranks do nothing here
        pass


def make_output_folder(Re, img_fname, channel_mesh_size, comm, rank):
    import os

    # Only rank 0 changes directories and creates folders
    if rank == 0:
        try:
            cwd = os.getcwd()

            # Safely remove suffix/prefix (compatible with older Python)
            if img_fname.endswith(".png"):
                img_name = img_fname[:-4]
            else:
                img_name = img_fname

            # Remove current working directory and "/InletImages/" prefix if present
            if img_name.startswith(cwd):
                img_name = img_name[len(cwd):]
            if img_name.startswith("/InletImages/"):
                img_name = img_name[len("/InletImages/"):]

            # Sanitize channel_mesh_size string
            channel_mesh_size_str = str(channel_mesh_size).replace(".", "")

            # Change directory once, check if exists
            noether_path = os.path.join(cwd, 'noether_data')
            if not os.path.isdir(noether_path):
                os.makedirs(noether_path)
            os.chdir(noether_path)

            folder_name = f'NSChannelFlow_RE{Re}_MeshLC{channel_mesh_size_str}_{img_name}'
            
            # Create folder if not exists
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

            # Return full folder path relative to cwd
            full_folder_path = os.path.join(noether_path, folder_name)

        except Exception as e:
            print(f"[Rank {rank}] ERROR in make_output_folder: {e}")
            raise
    else:
        full_folder_path = None
        img_name = None

    # Broadcast folder path and img_name to all ranks
    full_folder_path = comm.bcast(full_folder_path, root=0)
    img_name = comm.bcast(img_name, root=0)

    return full_folder_path, img_name


def solve_NS_flow():
    """
    Solves the incompressible Navier-Stokes flow in a domain derived from an input image.

    This function performs the following steps:
        1. Parses simulation parameters from user input or command line.
        2. Generates inlet velocity profiles by solving the Stokes problem.
        3. Solves a coarse Navier-Stokes problem using an intermediate mesh.
        4. Refines the mesh and solves the full Navier-Stokes equations using the 
           previously computed solution as an initial guess.
        5. Extracts the velocity solution and corresponding spatial coordinates.

    Returns:
        msh (Mesh): Final mesh used for the full Navier-Stokes solution.
        uh (Function): Placeholder velocity function (not fully used here), it is a Dolfinx function.
        uvw_data (np.ndarray): Velocity vector at unique degrees of freedom.
        xyz_data (np.ndarray): Corresponding spatial coordinates of the velocity vectors.
        Re (int): Reynolds number in the simulation
        img_fname (str): Full path to the input image file (should end with '.png').
        channel_mesh_size (float): Mesh size for the channel flow simulation (1 is defined as the width of the channel)

    Notes:
        - This function assumes MPI parallel execution and uses rank information
          for process-specific logic (though not shown in detail here).
        - The input image is used to generate the computational domain.
        - The simulation is Reynolds number dependent and may involve nonlinear solves.
        - This function is used with the "streamtrace.py" file into the file "INSERT NAME" to solve and streamtrace images in a batch file

    Dependencies:
        - Requires MPI (via mpi4py) and numerical solvers like PETSc/SNES.
        - External helper functions: `parse_arguments`, `generate_inlet_profiles`, 
          `generate_mesh`, `define_function_spaces`, `create_boundary_conditions`, 
          `setup_stokes_weak_form`, `solve_stokes_problem`, 
          `define_navier_stokes_form`, `solve_navier_stokes`.

    Assumptions:
        - The image file provided can be converted into a valid mesh.
        - The mesh and function spaces are compatible with FEniCSx or similar framework.
        - The function `snes_ksp_type` is defined or passed in the global scope.
    """
    # Get Inputs
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    Re, img_fname, flowrate_ratio, channel_mesh_size = parse_arguments()

    # Solve Stokes Flow
    uh_1, msh_1, uh_2, msh_2 = generate_inlet_profiles(img_fname, flowrate_ratio)
    msh, ft = generate_mesh(img_fname, 0.1)
    V, Q = define_function_spaces(msh)
    W, bcs = create_boundary_conditions(msh, ft, V, Q, uh_1, uh_2)
    a, L, V = setup_stokes_weak_form(W, msh)
    U_stokes = solve_stokes_problem(a, L, bcs, W)

    # Solve Coarse Navier Stokes
    a, w, dF, V = define_navier_stokes_form(W, msh, Re, U_stokes = U_stokes)
    w_coarse, u, p = solve_navier_stokes(a, w, dF, bcs, W, snes_ksp_type, comm, rank)

    # Solve Navier Stokes With User Defined Mesh
    msh, ft = generate_mesh(img_fname, channel_mesh_size)
    V, Q = define_function_spaces(msh)
    W, bcs = create_boundary_conditions(msh, ft, V, Q, uh_1, uh_2)
    a, w, dF, V = define_navier_stokes_form(W, msh, Re, U_coarse = w_coarse)
    w, u, p = solve_navier_stokes(a, w, dF, bcs, W, snes_ksp_type, comm, rank)

    uh = Function(V)
    dof_coords = V.tabulate_dof_coordinates()[:,:3]
    # Find number of components
    element = V.ufl_element()
    try:
        n_comp = element.value_shape()[0]
    except AttributeError:
        # If it's a blocked element, assume each sub-element is scalar.
        n_comp = len(element.sub_elements)

    # Reshape function values based on the number of components.
    values = u.x.array.reshape(-1, n_comp)

    # Extract unique vertex coordinates.
    xyz_data, unique_indices = np.unique(dof_coords, axis=0, return_index=True)
    uvw_data = values[unique_indices]

    return msh, uh, uvw_data, xyz_data, Re, img_fname, channel_mesh_size, V, Q, flowrate_ratio, u, p

def main():
    # Get Inputs
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    Re, img_fname, flowrate_ratio, channel_mesh_size = parse_arguments()
    folder_name, img_name = make_output_folder(Re, img_fname, channel_mesh_size)

    # Solve Stokes Flow
    uh_1, msh_1, uh_2, msh_2 = generate_inlet_profiles(img_fname, flowrate_ratio)
    msh, ft = generate_mesh(img_fname, 0.1)
    V, Q = define_function_spaces(msh)
    W, bcs = create_boundary_conditions(msh, ft, V, Q, uh_1, uh_2)
    a, L, V = setup_stokes_weak_form(W, msh)
    U_stokes = solve_stokes_problem(a, L, bcs, W)

    # Solve Coarse Navier Stokes
    a, w, dF, V = define_navier_stokes_form(W, msh, 1, U_stokes = U_stokes)
    w_coarse, u, p = solve_navier_stokes(a, w, dF, bcs, W, snes_ksp_type, comm, rank)

    # Solve Navier Stokes With User Defined Mesh
    msh, ft = generate_mesh(img_fname, channel_mesh_size)
    V, Q = define_function_spaces(msh)
    W, bcs = create_boundary_conditions(msh, ft, V, Q, uh_1, uh_2)
    a, w, dF, V = define_navier_stokes_form(W, msh, Re, U_coarse = w_coarse)
    w, u, p = solve_navier_stokes(a, w, dF, bcs, W, snes_ksp_type, comm, rank)

    save_navier_stokes_solution(u, p, msh, folder_name, Re)
    write_run_metadata(folder_name, Re, img_fname, flowrate_ratio, channel_mesh_size, V, Q, img_name)

    return w, W, msh

if __name__ == "__main__":
    main()