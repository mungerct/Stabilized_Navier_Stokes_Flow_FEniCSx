#!/usr/bin/env python 
# image 2 
import time
import gmsh
import numpy as np
from PIL import Image

from skimage import io
import skimage as sk
import scipy.ndimage as ndimage
import math
from rdp import rdp

# fenicsx solver
from dolfinx import fem
from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
from petsc4py import PETSc
import ufl

from dolfinx.io import gmshio
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

import sys
import os

# Visualization
import pyvista
from dolfinx.plot import vtk_mesh

# Inlet processing
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


comm = MPI.COMM_WORLD
rank = comm.Get_rank()



def load_image(img_fname): # Want this entire function
    #print('Loading image {}'.format(img_fname))
    img = sk.io.imread(img_fname)

    # print(img.shape)
    if (len(img.shape) == 2):
        gray_img = img
    else:
        if (img.shape[2] == 3):
            gray_img = sk.color.rgb2gray(img)
        if (img.shape[2] == 4):
            rgb_img = sk.color.rgba2rgb(img)
            gray_img = sk.color.rgb2gray(rgb_img)

    return gray_img

def get_contours(gray_img): # Want this entire function
    height, width = gray_img.shape    
    # Normalize and flip (for some reason)
    raw_contours = sk.measure.find_contours(gray_img, 0.5) # Start with this, NOT the optimized contours
 
    #print('Found {} contours'.format(len(raw_contours)))

    contours = []
    for n, contour in enumerate(raw_contours):
        # Create an empty image to store the masked array
        r_mask = np.zeros_like(gray_img, dtype = int)  # original np.int
        # Create a contour image by using the contour coordinates rounded to their nearest integer value
        r_mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
        # Fill in the hole created by the contour boundary
        r_mask = ndimage.binary_fill_holes(r_mask)

        contour_area = float(np.count_nonzero(r_mask))/(float(height * width))
        #print(np.count_nonzero(r_mask))
        if (contour_area >= 0.05):
            contours.append(contour)

    #print('Reduced to {} contours'.format(len(contours)))

    for n, contour in enumerate(contours):
        contour[:,1] -= 0.5 * height
        contour[:,1] /= height

        contour[:,0] -= 0.5 * width
        contour[:,0] /= width
        contour[:,0] *= -1.0

    #print("{:d} Contours detected".format(len(contours)))

    return contours


def optimize_contour(contour):
    #print("Optimizing contour.")
    dir_flag = 0
    dir_bank = []

    contour_keep = []

    ## Use low-pass fft to smooth out 
    x = contour[:,1]
    y = contour[:,0]

    signal = x + 1j*y
    #print(signal)

    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1])
    cutoff = 0.12
    fft[np.abs(freq) > cutoff] = 0 

    signal_filt = np.fft.ifft(fft)

    contour[:,1] = signal_filt.real
    contour[:,0] = signal_filt.imag

    #contour = rdp(contour)
    contour = rdp(contour, epsilon=0.0005)

    # Remove final point in RDP, which coincides with
    # the first point
    contour = np.delete(contour, len(contour)-1, 0)

    # cutoff of 0.15, eps of 0.005 works for inner flow

    #contour = reverse_opt_pass(contour)
    # Figure out a reasonable radius
    max_x = max(contour[:,1])
    min_x = min(contour[:,1])
    
    max_y = max(contour[:,0])
    min_y = min(contour[:,0])
    
    # Set characteristic lengths, epsilon cutoff
    lc = min((max_x - min_x), (max_y - min_y))
    mesh_lc = 0.01 * lc    

    return [contour, mesh_lc]

def outer_contour_to_gmsh(contour, mesh_lc, p_idx=1, l_idx=1, loop_idx=1):
    #print('Running outer_contour_to_gmsh')
    line_init = l_idx
    g = gmsh.model.geo
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",0)
    gmsh.model.add("outer_contour_mesh")
    lc = mesh_lc
    #lc = 0.1 # comment out this line
    g.addPoint(0, -0.5, -0.5, lc, p_idx)
    g.addPoint(0, 0.5, -0.5, lc, p_idx + 1)
    g.addPoint(0, 0.5,  0.5, lc, p_idx + 2)
    g.addPoint(0, -0.5,  0.5, lc, p_idx + 3)
    
    g.addLine(p_idx, p_idx + 1, l_idx)
    g.addLine(p_idx + 1, p_idx + 2, l_idx + 1)
    g.addLine(p_idx + 2, p_idx + 3, l_idx + 2)
    g.addLine(p_idx + 3, p_idx, l_idx + 3)
    g.addCurveLoop(list(range(l_idx , l_idx + 4)), loop_idx)
    
    p_idx += 3
    p_idx_closure = p_idx + 1
    for point in contour:
        p_idx += 1
        g.addPoint(0, point[1], point[0], lc, p_idx)
    
    l_idx += 3
    for line in range(len(contour)-1):
        l_idx += 1
        g.addLine(l_idx, l_idx + 1, l_idx)
        
    g.addLine(l_idx + 1, p_idx_closure, l_idx + 1)
    g.addCurveLoop(list(range(p_idx_closure , l_idx + 2)), 2)
    g.addPlaneSurface([1,2], 1)
    g.synchronize()
    
    gmsh.model.addPhysicalGroup(1, list(range(1 , l_idx + 2)), name = "walls")
    gmsh.model.addPhysicalGroup(2, [1], name = "outer_surface") 
    
    
    
    gmsh.model.mesh.generate(2)
    gmsh.write("outer_contour_mesh.msh")
    gmsh.write('outer_contour.geo_unrolled')

    if rank == 0:
        print('Saved the Outer Contour Mesh', flush = True)
 
    return gmsh.model

def inner_contour_to_gmsh(contour, mesh_lc):
    #print('Running inner_contour_to_gmsh')
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",0)
    gmsh.model.add("inner_contour_mesh")
    g = gmsh.model.geo
    lc = mesh_lc
    #lc = 6  # comment out this line
    idx = 0
    for point in contour:
        idx += 1
        g.addPoint(0, point[1], point[0], lc, idx)
    idx = 0
    for line in range(len(contour)-1):
        idx += 1
        g.addLine(idx, idx + 1, idx)
    g.addLine(idx + 1, 1, idx + 1)
    g.addCurveLoop(list(range(1 , idx + 2)), 1)
    g.addPlaneSurface([1], 1)
    g.synchronize()
    gmsh.model.addPhysicalGroup(1, list(range(1 , idx + 2)), name = "walls")
    gmsh.model.addPhysicalGroup(2, [1], name = "inner_surface") 
    gmsh.model.mesh.generate(2)
    gmsh.write("inner_contour_mesh.msh")
    gmsh.write('inner_contour.geo_unrolled')
    if rank == 0:
        print("Saved the Inner Contour Mesh", flush = True)
    
    return gmsh.model

def process_2_channel_mesh_model(contours):
    contour_inner = contours[1]
    contour_outer = contours[0]

    [contour_inner, mesh_lc_a] = optimize_contour(contour_inner) # Want this output for streamtracing (contour_inner) should be a collection of x-y-z coords
    [contour_outer, mesh_lc_c] = optimize_contour(contour_outer)

    inner_shape = create_inner_shape(contour_inner)

    inner_model = inner_contour_to_gmsh(contour_inner, mesh_lc_a)
    outer_model = outer_contour_to_gmsh(contour_outer, mesh_lc_c)
    
    return inner_model, outer_model, inner_shape

def image2gmsh(img_fname):
    img = load_image(img_fname)
    contours = get_contours(img)
    inner_model, outer_model, inner_shape = process_2_channel_mesh_model(contours)

    return inner_model, outer_model, inner_shape

def solve_velocity_field(gmsh_model):
    gdim = 3
    gmsh_model_rank = 0
    mesh_comm = MPI.COMM_WORLD
    msh, cell_markers, facet_markers = gmshio.model_to_mesh(gmsh_model, mesh_comm, gmsh_model_rank, gdim=gdim)

    V = fem.functionspace(msh, ("Lagrange", 1))
    
    # Defining an arbitrary forcing function (pressure gradient dp/dx, will normalize all
    # flow to average = 1.0 anyway)
    p = 10
    
    # Find mesh area area
    one = fem.Constant(msh, PETSc.ScalarType(1))
    f = fem.form(one*ufl.dx)
    area = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(f))
    #print(f'Area = {area}')

    # Outer no-slip
    noslip = fem.Constant(msh, PETSc.ScalarType(0))
    dofs = fem.locate_dofs_topological(V, msh.topology.dim-1, facet_markers.find(1))
    bc = fem.dirichletbc(noslip, dofs, V)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = p * v * ufl.dx
    problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    
    # Compute average velocity
    f2 = fem.form(uh*ufl.dx)
    average_velocity = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(f2))/area
    #print(f'Mean velocity = {average_velocity}')

    
    # Visualization with PyVista
    #topology, cell_types, x = vtk_mesh(msh, msh.topology.dim)
    #grid = pyvista.UnstructuredGrid(topology, cell_types, x)
    #grid.point_data["u"] = uh.x.array
    #warped = grid.warp_by_scalar("u", factor= 0.25 )
    #plotter = pyvista.Plotter()
    #plotter.background_color = "white"
    #plotter.add_mesh(warped, show_edges=False, show_scalar_bar=False, scalars="u")
    #plotter.show_bounds()
    #if not pyvista.OFF_SCREEN:
    #    plotter.show()
    #else:
    #    plotter.screenshot("deflection.png") 

    return uh, area, average_velocity, msh, V

def solve_inlet_profiles(img_fname, flowrate_ratio):
    # Convert image to meshes and inner shape image
    inner_model, outer_model, inner_shape = image2gmsh(img_fname)

    # Get inner and outer flow profiles
    inner_model.setCurrent("inner_contour_mesh")
    uh_1, area_1, average_u_1, msh_1, V_1 = solve_velocity_field(inner_model)
    outer_model.setCurrent("outer_contour_mesh")
    uh_2, area_2, average_u_2, msh_2, V_2 = solve_velocity_field(outer_model)

    # Normalize both velocity profiles by average to get average = 1.0
    uh_1.x.array[:] = uh_1.x.array[:] / average_u_1
    uh_2.x.array[:] = uh_2.x.array[:] / average_u_2

    # Confirm average = 1.0
    #f1 = fem.form(uh_1*ufl.dx)
    #average_velocity_1 = fem.assemble_scalar(f1)/area_1
    #f2 = fem.form(uh_2*ufl.dx)
    #average_velocity_2 = fem.assemble_scalar(f2)/area_2
    
    # Determine scalar multiplier to adjust velocities based on
    # flowrate ratio and each flow area
    flow_u_1 = flowrate_ratio / area_1
    flow_u_2 = (1.0 - flowrate_ratio) / area_2

    # Scale velocity fields
    uh_1.x.array[:] = uh_1.x.array[:] * flow_u_1
    uh_2.x.array[:] = uh_2.x.array[:] * flow_u_2

    # Confirm new average velocity
    #f1 = fem.form(uh_1*ufl.dx)
    #average_velocity_1 = fem.assemble_scalar(f1)/area_1
    #f2 = fem.form(uh_2*ufl.dx)
    #average_velocity_2 = fem.assemble_scalar(f2)/area_2


    # Get coordinates of each velocity field to interpolate onto common grid
    #coor_1 = V_1.tabulate_dof_coordinates()
    #u_1 = uh_1.x.array.real.astype(np.float32)
    #coor_2 = V_2.tabulate_dof_coordinates()
    #u_2 = uh_2.x.array.real.astype(np.float32)

    return uh_1, msh_1, uh_2, msh_2

def create_inner_shape(contour_points):
    # Fill image with inner flow shape

    polygon = Polygon(contour_points)

    nx = 256
    ny = 256

    y_min = -0.5
    y_max = 0.5
    x_min = -0.5
    x_max = 0.5

    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)

    grid = np.zeros((nx,ny), dtype=np.uint8)

    for j in range(0, ny):
        for i in range(0, nx):
            point = Point(x[i], y[j])

            if polygon.contains(point):
                grid[i,j] = 255
                #f.write('1\n')
            else:
                grid[i,j] = 0
                #f.write('0\n')

    # img = Image.fromarray(grid, 'L')
    # img.save('inner_shape.png')

    return grid
    
def main(img_fname, flowrate_ratio):
    uh_1, msh_1, uh_2, msh_2 = solve_inlet_profiles(img_fname, flowrate_ratio)
    return uh_1, msh_1, uh_2, msh_2

if __name__ == '__main__':
    img_fname = sys.argv[1]
    flowrate_ratio = float(sys.argv[2])
    main(img_fname, flowrate_ratio)