#!/usr/bin/env python
import time
import dolfinx.fem.function
import numpy as np

from skimage import io
import skimage as sk
import scipy.ndimage as ndimage
from scipy.ndimage import binary_dilation
from rdp import rdp

import sys
import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# Inlet processing
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import dolfinx
import dolfinx.io
import dolfinx.fem
import ufl
from mpi4py import MPI
import sys
import h5py
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
from dolfinx.io import XDMFFile
import basix
import adios4dolfinx
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from dolfinx import geometry
from scipy.integrate import solve_ivp
from image2inlet import solve_inlet_profiles, optimize_contour, get_contours, load_image
import alphashape
from descartes import PolygonPatch
from multiprocessing import Process, Queue
from multiprocessing import Pool, cpu_count
from functools import partial
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import cpu_count

comm = MPI.COMM_WORLD
global bb_tree
global mesh
global uh
global uvw_data
global xyz_data

def pause():
    # Define pause function for debugging
    programPause = input("Press the <ENTER> key to continue...")

def read_mesh_and_function(fname_base, function_name, function_dim):
    print('Reading solution from file', flush = True)
    '''
    INPUTS
    fname_base:     file prefix, e.g., for data_u.xdmf, fname_base = data_u
    function_name:  name of function saved to xdmf file, e.g., 'Velocity'
    function_dime:  number of dimensions in function space, e.g., 2D velocity field: 2
    
    OUTPUTS
    mesh:           mesh from saved data
    uh:             function from saved data (Dolfinx velocity function)
    data:           raw numpy array of saved data
    '''
    # Read in mesh file
    with XDMFFile(MPI.COMM_SELF, f"{fname_base}.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="mesh")

    num_nodes_global = mesh.geometry.index_map().size_global

    # Create the function space and function
    P2 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    V = dolfinx.fem.functionspace(mesh, P2)

    xyz = V.tabulate_dof_coordinates()

    uh = dolfinx.fem.Function(V)

    h5_filename = f"{fname_base}.h5"
    with h5py.File(h5_filename, "r") as h5f:
        #print("Datasets in HDF5 file:", list(h5f.keys()), flush = True)
        # print("Data keys in the 'Function' group:", list(h5f["Function"].keys()))
        func_group = h5f["Function"]
        #print("Keys in 'Function':", list(func_group.keys()), flush = True)

        velocity_group = func_group[function_name]
        # print(f"Keys in '{function_name}':", list(velocity_group.keys()), flush = True)
        
        data = h5f["Function"][function_name]["0"][...]

    local_input_range = adios4dolfinx.comm_helpers.compute_local_range(mesh.comm, num_nodes_global)
    local_input_data = data[local_input_range[0]:local_input_range[1]]

    shape = data.shape

    x_dofmap = mesh.geometry.dofmap
    igi = np.array(mesh.geometry.input_global_indices, dtype=np.int64)
    global_geom_input = igi[x_dofmap]
    global_geom_owner = adios4dolfinx.utils.index_owner(mesh.comm, global_geom_input.reshape(-1), num_nodes_global)
    for i in range(function_dim):
        arr_i = adios4dolfinx.comm_helpers.send_dofs_and_recv_values(global_geom_input.reshape(-1), global_geom_owner, mesh.comm, local_input_data[:,i], local_input_range[0])
        dof_pos = x_dofmap.reshape(-1)*function_dim+i
        uh.x.array[dof_pos] = arr_i

    # Find number of components
    element = V.ufl_element()
    try:
        n_comp = element.value_shape()[0]
    except AttributeError:
        # If it's a blocked element, assume each sub-element is scalar.
        n_comp = len(element.sub_elements)

    # Get dof coordinates from the function space.
    dof_coords = V.tabulate_dof_coordinates()[:,:function_dim]

    # Reshape function values based on the number of components.
    values = uh.x.array.reshape(-1, n_comp)

    # Extract unique vertex coordinates.
    xyz_data, unique_indices = np.unique(dof_coords, axis=0, return_index=True)
    uvw_data = values[unique_indices]

    return mesh, uh, uvw_data, xyz_data

def update_contour(img_fname):
    # This function takes in the image filename and prepares it to be streamtraced
    print('Finding Image Contour', flush = True)
    gray_img = load_image(img_fname)
    img_contours = get_contours(gray_img)
    contour, mesh_lc = optimize_contour(img_contours[1])
    zeros_col = np.zeros((contour.shape[0], 1))
    contour[:, [0,1]] = contour[:, [1,0]]
    new_arr = np.hstack((zeros_col, contour))
    return new_arr

def velfunc(t, x, bb_tree, mesh, uh):
    # This is the velocity function, it finds the velocity at a given point in the domain
    cell_candidate = geometry.compute_collisions_points(bb_tree, x) # Choose one of the cells that contains the point
    colliding_cell = geometry.compute_colliding_cells(mesh, cell_candidate, x) # Choose one of the cells that contains the point
    if len(colliding_cell.links(0)) == 0:
        # If the point is outside of the domain, set its velocity to be zero
        # print("Point Outside Domain", flush = True)
        vel = np.array([0, 0, 0])
        return vel
    else:
        cell_index = colliding_cell.links(0)[0]
        vel = uh.eval(x, [cell_index])
        # print(f'P:{x}, V:{vel}', flush = True)
        return vel

def velfunc_reverese(t, x, bb_tree, mesh, uh):
    # This is the velocity function, it finds the velocity at a given point in the domain
    cell_candidate = geometry.compute_collisions_points(bb_tree, x) # Choose one of the cells that contains the point
    colliding_cell = geometry.compute_colliding_cells(mesh, cell_candidate, x) # Choose one of the cells that contains the point
    if len(colliding_cell.links(0)) == 0:
        # print(f"[DEBUG] Point {x} is outside the mesh — returning zero velocity", flush=True)
        # If the point is outside of the domain, set its velocity to be zero
        vel = np.array([0, 0, 0])
        return vel
    else:
        cell_index = colliding_cell.links(0)[0]
        vel = uh.eval(x, [cell_index])
        vel = vel*(-1)
        # print(f'P:{x}, V:{vel}', flush = True)
        return vel

def velocity_magnitude_event(t, y, bb_tree, mesh, uh):
    # Event flag for the "solve_ivp" function from Scipy, triggers if the particle stops moving
    speed = np.linalg.norm(velfunc(t, y, bb_tree, mesh, uh))
    return speed - 1e-6  # triggers when speed is 1e-6

def position_event(t, y, bb_tree, mesh, uh):
    # Event flag for the "solve_ivp" function from Scipy, triggers when the particle is at x = 3.7 (the total domain is length = 4)
    pos_x = y[0]
    return pos_x - 3.7 # triggers when x is at 3.7

def reverse_position_event(t, y, bb_tree, mesh, uh):
    # Event flag for the "solve_ivp" function from Scipy, triggers when the particle is at x = 3.7 (the total domain is length = 4)
    pos_x = y[0]
    return pos_x - 0.13 # triggers when x is at 0.06

def inner_contour_mesh_func(img_fname):
    # Make a mesh of the inner countor and used those points to streamtrace
    inner_mesh = solve_inlet_profiles(img_fname, 0.5)[1]
    print("Made inner mesh", flush=True)
    inner_mesh = inner_mesh.geometry.x
    return inner_mesh

def streamtrace_pool(row, bb_tree, mesh, uh):
    t_span = (0, 20)
    velocity_magnitude_event.terminal = True  # stops integration when event is triggered
    velocity_magnitude_event.direction = -1   # only when crossing threshold from above
    position_event.terminal = True
    position_event.direction = 1
    events_list = (velocity_magnitude_event, position_event)

    sol = solve_ivp(velfunc, t_span, row, method='RK45', events=events_list, max_step=0.125, args=(bb_tree, mesh, uh))
    x_vals = np.array(sol.y[0])
    y_vals = np.array(sol.y[1])
    z_vals = np.array(sol.y[2])

    if x_vals[-1] > 0.5:
        return (
            [x_vals[-1]],
            [y_vals[-1]],
            [z_vals[-1]],
        )
    else:
        return None

def run_streamtrace(inner_mesh, bb_tree, mesh, uh):
    start_time = time.time()
    print('Streamtracing', flush=True)

    wrapped_streamtrace = partial(streamtrace_pool, bb_tree=bb_tree, mesh=mesh, uh=uh)
    
    with ThreadPool(processes=cpu_count()) as pool:
        results = pool.map(wrapped_streamtrace, [inner_mesh[i, :] for i in range(inner_mesh.shape[0])])

    # print("Sample results:")
    # for i, res in enumerate(results):
        # print(f"[{i}] type={type(res)}, value={res}")

    # Filter out None results
    results = [res for res in results if res is not None]

    if results:
        pointsx, pointsy, pointsz = zip(*results)
        pointsx = np.array(pointsx)
        pointsy = np.array(pointsy)
        pointsz = np.array(pointsz)
    else:
        pointsx = np.array([])
        pointsy = np.array([])
        pointsz = np.array([])

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds", flush = True)
    return pointsx, pointsy, pointsz

def plot_streamtrace(pointsy, pointsz, contour, limits):
    pointsy = np.squeeze(pointsy)
    pointsz = np.squeeze(pointsz)

    points = np.vstack((pointsy, pointsz))
    points = points.T

    alpha_shape = alphashape.alphashape(points, 0.2)
    # Initialize plot
    fig, ax = plt.subplots()
    x = np.array(list(alpha_shape.exterior.coords)).T[0]
    y = np.array(list(alpha_shape.exterior.coords)).T[1]

    # plt.scatter(x, y, marker = '.', color = 'b')
    plt.fill(x, y)
    plt.gca().set_aspect('equal')
    ax.set_xlim(-1*limits, limits)
    ax.set_ylim(-1*limits, limits)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('Alpha Shape')
    plt.show()

    plt.scatter(pointsy, pointsz, marker = 'o') # Make stream trace outlet profile
    plt.gca().set_aspect('equal')
    ax.set_xlim(-1*limits, limits)
    ax.set_ylim(-1*limits, limits)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('Scatter Plot')
    plt.show()

    return(plt)

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

def expand_streamtace(pointsy, pointsz, contour):
    print('Expanding edges of forward streamtrace')
    pointsy = np.squeeze(pointsy)
    pointsz = np.squeeze(pointsz)

    points = np.vstack((pointsy, pointsz)).T

    alpha_shape = alphashape.alphashape(points, 0.2)

    if isinstance(alpha_shape, Polygon):
        polygon = alpha_shape
    elif isinstance(alpha_shape, (MultiPolygon, GeometryCollection)):
        # Extract polygons and pick the largest by area
        polygons = [geom for geom in alpha_shape.geoms if isinstance(geom, Polygon)]
        if not polygons:
            raise ValueError("No Polygon found inside alpha shape GeometryCollection or MultiPolygon.")
        polygon = max(polygons, key=lambda p: p.area)
    else:
        raise ValueError(f"Alpha shape is not a Polygon. Got: {type(alpha_shape)}. "
                         "This may indicate a degenerate point set or bad alpha parameter.")

    x = np.array(list(polygon.exterior.coords)).T[0]
    y = np.array(list(polygon.exterior.coords)).T[1]

    blurr = 0.2

    # Move the min/max x values "out" to cast a refined reverse streamtrace
    if min(x) <= 0 and max(x) >= 0:
        min_index = np.argmin(x)
        x[min_index] = -1*abs(x[min_index]*blurr) + -1*abs(x[min_index])
        max_index = np.argmax(x)
        x[max_index] = x[max_index]*blurr + x[max_index]
    else:
        min_index = np.argmin(x)
        x[min_index] = -1*x[min_index]*blurr +  x[min_index]
        max_index = np.argmax(x)
        x[max_index] = x[max_index]*blurr + x[max_index]

    # Move the min/max y values "out" to cast a refined reverse streamtrace
    if min(y) <= 0 and max(y) >= 0:
        min_index = np.argmin(y)
        y[min_index] = -1*abs(y[min_index]*blurr) + -1*abs(y[min_index])
        max_index = np.argmax(y)
        y[max_index] = y[max_index]*blurr + y[max_index]
    else:
        min_index = np.argmin(y)
        y[min_index] = -1*y[min_index]*blurr +  y[min_index]
        max_index = np.argmax(y)
        y[max_index] = y[max_index]*blurr + y[max_index]

    return min(x), max(x), min(y), max(y)


def make_rev_streamtrace_seeds(minx, maxx, miny, maxy, numpoints):
    x = np.linspace(minx, maxx, num = numpoints)
    y = np.linspace(miny, maxy, num = numpoints)
    x, y = np.meshgrid(x, y)
    points = np.stack((x, y), axis=-1)
    array = points.reshape(-1, 2)
    fours_col = np.ones((array.shape[0], 1))*3.9
    new_arr = np.hstack((fours_col, array))

    return new_arr # Array of new seeds for reverse stream trace

def reverse_streamtrace_pool(row, bb_tree, mesh, uh):
    t_span = (0, 20)
    velocity_magnitude_event.terminal = True
    velocity_magnitude_event.direction = -1
    reverse_position_event.terminal = True
    reverse_position_event.direction = -1
    events_list = (reverse_position_event, velocity_magnitude_event)

    sol = solve_ivp(velfunc_reverese, t_span, row, method='RK45', events=events_list, max_step=0.125, args=(bb_tree, mesh, uh))

    x_vals = np.array(sol.y[0])
    # print(x_vals, flush=True)
    y_vals = np.array(sol.y[1])
    z_vals = np.array(sol.y[2])

    if x_vals[-1] < 0.5:
        return (
            [x_vals[-1]],
            [y_vals[-1]],
            [z_vals[-1]]
        )
    else:
        return (
            [10],
            [10],
            [10]
        )

def run_reverse_streamtrace(seeds, mesh, uh):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"[Rank {rank}] Building bb_tree locally", flush=True)
    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)

    # Step 1: Broadcast shared data
    if rank == 0:
        print('Reverse Streamtracing with MPI', flush=True)
        start_time = time.time()

    # Step 2: Split seeds across ranks
    num_seeds = seeds.shape[0] if rank == 0 else None
    num_seeds = comm.bcast(num_seeds, root=0)

    # Scatter seeds
    counts = [num_seeds // size + (1 if i < num_seeds % size else 0) for i in range(size)]
    displs = np.cumsum([0] + counts[:-1])
    local_count = counts[rank]

    local_seeds = np.empty((local_count, seeds.shape[1]), dtype=seeds.dtype)
    comm.Scatterv(
        [seeds, tuple(np.array(counts) * seeds.shape[1]), tuple(displs * seeds.shape[1]), MPI.DOUBLE],
        local_seeds,
        root=0
    )

    # Step 3: Each rank computes local results
    local_results = []
    for seed in local_seeds:
        res = reverse_streamtrace_pool(seed, bb_tree=bb_tree, mesh=mesh, uh=uh)
        local_results.append(res if res is not None else (np.nan, np.nan, np.nan))

    # Step 4: Gather all results back to root
    local_results_np = np.array(local_results, dtype=np.float64)  # shape: (local_count, 3)
    recvbuf = None
    if rank == 0:
        recvbuf = np.empty((num_seeds, 3), dtype=np.float64)

    comm.Gatherv(
        local_results_np,
        [recvbuf, (np.array(counts) * 3, displs * 3), MPI.DOUBLE],
        root=0
    )

    # Step 5: Finalize results on root
    if rank == 0:
        pointsx = recvbuf[:, 0]
        pointsy = recvbuf[:, 1]
        pointsz = recvbuf[:, 2]

        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time:.4f} seconds", flush=True)
        return pointsx, pointsy, pointsz
    else:
        return None, None, None

def find_seed_end(rev_pointsy, rev_pointsz, seeds, contour):
    contour = contour[:, 1:3]
    # contour[:,[1,0]] = contour[:,[0,1]]
    valid_seeds = []

    for i in range(seeds.shape[0]):
        point = np.array([rev_pointsy[i], rev_pointsz[i]])
        point = point.reshape(1, 2)
        is_inside = sk.measure.points_in_poly(point, contour)

        if is_inside[0]: # if the point is inside the contour
            valid_seeds.append(seeds[i])
    
    valid_seeds = np.array(valid_seeds)

    valid_seeds = valid_seeds[:, 1:3]

    return valid_seeds

def plot_inlet(contour, inner_mesh, limits):
    print('Plotting Inlet Contour and Mesh', flush = True)
    inner_contour_fig, ax = plt.subplots() 
    ax.fill(contour[:,1],contour[:,2])
    ax.set_aspect('equal')
    ax.set_xlim(-1*limits, limits)
    ax.set_ylim(-1*limits, limits)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('Inner Contour')

    inner_contour_mesh_fig, ax = plt.subplots()
    ax.scatter(inner_mesh[:,1], inner_mesh[:,2])
    ax.set_aspect('equal')
    ax.set_xlim(-1*limits, limits)
    ax.set_ylim(-1*limits, limits)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('Inner Contour Mesh')

    return inner_contour_fig, inner_contour_mesh_fig

def parse_arguments():
    print(len(sys.argv), flush = True)
    if len(sys.argv) not in [4]:
        raise ValueError("Usage: script.py <img_fname> <solname> <funcname>")
    img_fname = sys.argv[1] # File name of input image
    solname = sys.argv[2] # base name of .xdmf file (test.xdmf is just test)
    funcname = sys.argv[3] # Name of function ("Velocity" or "Pressure", etc.)
    funcdim = 3 # Dimension of solution (2 or 3)

    print("Accepted Inputs", flush = True)
    num_cpus = cpu_count()
    print(f"Number of CPUs: {num_cpus}", flush = True)

    return img_fname, solname, funcname, funcdim

def move_directory(img_fname):
    curr_dir = os.getcwd()
    folder = os.path.dirname(img_fname)
    os.chdir(folder)

def save_figs(img_fname, inner_contour_fig, inner_contour_mesh_fig, seeds, final_output, rev_streamtrace_fig, num_seeds):
    move_directory(img_fname)
    print('Saving Figures', flush = True)
    inner_contour_fig.savefig("inner_contour.svg")
    inner_contour_mesh_fig.savefig("inner_mesh.svg")
    print(img_fname, flush = True)
    img_fname = os.path.basename(img_fname)
    print(img_fname, flush = True)
    img_fname = img_fname.removesuffix(".png")
    print(img_fname, flush = True)
    rev_streamtrace_fig.savefig(f"rev_trace_{img_fname}_{num_seeds}.svg")
    np.savetxt("rev_seeds.csv", seeds, delimiter=",")
    np.savetxt("final_output.csv", final_output, delimiter=",")

def plot_rev_streamtrace(final_output, limits):
    print('Plotting Reverse Streamtrace', flush = True)

    rev_streamtrace_fig, ax = plt.subplots()
    ax.scatter(final_output[:, 0], final_output[:, 1], marker = ".")
    ax.set_aspect('equal')
    ax.set_xlim(-1*limits, limits)
    ax.set_ylim(-1*limits, limits)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # plt.show()

    return rev_streamtrace_fig

def for_and_rev_streamtrace(num_seeds, limits, img_fname, mesh, uh, uvw_data, xyz_data, Re, Folder_name):
    """
    Performs forward and reverse stream tracing based on an image-derived inlet contour and mesh data.

    Parameters:
        num_seeds (int): Number of seeds in the x and y direction to use for reverse stream tracing.
        limits (tuple): Plotting limits for visualization.
        img_fname (str): Filename of the input image used to extract the inlet contour.
        msh (Mesh): Finite element mesh of the domain.
        uh (Function): Velocity function, it is a Dolfinx function containing (x,y,z) velocity information
        uvw_data: Velocity field data (np array).
        xyz_data: Spatial coordinate data (np array).

    Workflow:
        1. Extract the inlet contour from the image file.
        2. Construct a bounding box tree from the mesh for spatial queries.
        3. Generate an inner mesh representing the inlet region.
        4. Plot and save visualizations of the inlet contour and its mesh.
        5. Run forward stream tracing on the inner mesh to generate flow paths.
        6. Calculate bounding box around the traced streamlines and create seed points for reverse tracing.
        7. Run reverse stream tracing using the generated seeds.
        8. Determine final streamline termination points from the reverse trace.
        9. Plot and save results of the reverse stream trace.

    Output:
        Saves three figures:
            - Inlet contour plot
            - Inlet mesh plot
            - Reverse streamtrace plot
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mesh, uh, uvw_data, xyz_data = read_mesh_and_function(f"{Folder_name}/Re{Re}ChannelVelocity", 'Velocity', 3)
    coords = mesh.geometry.x  # Nx3 array of vertex coords
    print(f"x range in mesh: {coords[:,0].min()} to {coords[:,0].max()}", flush=True)

    print(f"[Rank {rank}] Starting streamtrace function on {size} ranks", flush=True)
    if rank == 0:
        print(f"[Rank {rank}] STEP 1: Updating contour and building bb_tree", flush=True)
        contour = update_contour(img_fname)
        bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
    else:
        seeds = None
        bb_tree = None
        contour = None
        inner_contour_fig = None
        inner_contour_mesh_fig = None
    
    inner_mesh = inner_contour_mesh_func(img_fname)
    inner_mesh = comm.gather(inner_mesh, root=0)
    if rank == 0:
        # Combine list of arrays into one 2D array
        inner_mesh = np.vstack(inner_mesh)

    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
    contour = comm.bcast(contour, root=0)
    comm.Barrier()
    if rank == 0:
        print(f"[Rank {rank}] STEP 2: Plotting inlet stuff", flush=True)
        inner_contour_fig, inner_contour_mesh_fig = plot_inlet(contour, inner_mesh, limits)

        print(f"[Rank {rank}] STEP 3: Running forward streamtrace", flush=True)
        pointsx, pointsy, pointsz = run_streamtrace(inner_mesh, bb_tree, mesh, uh)

        print(f"[Rank {rank}] STEP 4: Expanding tracing region", flush=True)
        minx, maxx, miny, maxy = expand_streamtace(pointsy, pointsz, contour)

        print(f"[Rank {rank}] STEP 5: Generating seeds", flush=True)
        seeds = make_rev_streamtrace_seeds(minx, maxx, miny, maxy, num_seeds)

        # Create scatter plot
        plt.figure(figsize=(5, 5))
        plt.scatter(pointsy, pointsz, color='blue', marker='o', edgecolors='k')

        # Label axes and title
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        plt.title('Scatter Plot Example')
        plt.grid(True)

        # Show plot
        # plt.show()
    else:
        seeds = None
        bb_tree = None
        contour = None
        inner_contour_fig = None
        inner_contour_mesh_fig = None
    comm.Barrier()
    print(f"[Rank {rank}] STEP 6: Broadcasting seeds, bb_tree, and contour", flush=True)
    seeds = comm.bcast(seeds, root=0)
    comm.Barrier()
    print(f"[Rank {rank}] STEP 7: Running reverse streamtrace", flush=True)
    coords = mesh.geometry.x  # Nx3 array of vertex coords
    print(f"x range in mesh: {coords[:,0].min()} to {coords[:,0].max()}", flush=True)
    rev_pointsx, rev_pointsy, rev_pointsz = run_reverse_streamtrace(seeds, mesh, uh)

    if rank == 0:
        print(f"[Rank {rank}] STEP 8: Post-processing and plotting final output", flush=True)
        final_output = find_seed_end(rev_pointsy, rev_pointsz, seeds, contour)
        rev_streamtrace_fig = plot_rev_streamtrace(final_output, limits)

        print(f"[Rank {rank}] Finished streamtrace function", flush=True)
        return (
            rev_streamtrace_fig,
            inner_contour_fig,
            inner_contour_mesh_fig,
            seeds,
            final_output
        )
    else:
        print(f"[Rank {rank}] Finished streamtrace function (no output to return)", flush=True)
        return None, None, None, None, None


def main():
    limits = 0.5
    num_seeds = 50

    img_fname, solname, funcname, funcdim = parse_arguments()
    contour = update_contour(img_fname)

    mesh, uh, uvw_data, xyz_data = read_mesh_and_function(solname, funcname, funcdim)
    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
    inner_mesh = inner_contour_mesh_func(img_fname)

    inner_contour_fig, inner_contour_mesh_fig = plot_inlet(contour, inner_mesh, limits)

    pointsx, pointsy, pointsz = run_streamtrace(inner_mesh, bb_tree, mesh, uh)
    # plot_streamtrace(pointsy, pointsz, contour, limits)
    minx, maxx, miny, maxy = expand_streamtace(pointsy, pointsz, contour)
    seeds = make_rev_streamtrace_seeds(minx, maxx, miny, maxy, num_seeds)

    rev_pointsx, rev_pointsy, rev_pointsz = run_reverse_streamtrace(seeds, mesh, uh)
    final_output = find_seed_end(rev_pointsy, rev_pointsz, seeds, contour)

    # plot_streamtrace(rev_pointsy, rev_pointsz, contour, limits)
    rev_streamtrace_fig = plot_rev_streamtrace(final_output, limits)
    save_figs(img_fname, inner_contour_fig, inner_contour_mesh_fig, seeds, final_output, rev_streamtrace_fig, num_seeds)

if __name__ == "__main__":
    main()