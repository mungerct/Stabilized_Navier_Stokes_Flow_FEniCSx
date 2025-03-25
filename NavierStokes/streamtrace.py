#!/usr/bin/env python 
# image 2 
import time
import dolfinx.fem.function
import numpy as np

from skimage import io
import skimage as sk
import scipy.ndimage as ndimage
from rdp import rdp

import sys
import os

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

comm = MPI.COMM_WORLD
img_fname = sys.argv[1] # File name of input image
solname = sys.argv[2]
solname_h5 = sys.argv[3] # File name of solution (.xdmf file)

def load_image(img_fname):
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

def get_contours(gray_img):
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
        # contour[:,0] *= -1.0

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

def read_sol_file(comm, sol_fname_xdmf, sol_fname_h5):
    with dolfinx.io.XDMFFile(comm, sol_fname_xdmf, "r") as infile:
        # Read the mesh
        mesh = infile.read_mesh(name="mesh")
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

        # Create a function space (this needs to match the space used in the saved file)
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

        # Create a function in the space
        u = dolfinx.fem.Function(V)

        # --- Load function values from HDF5 file ---
    with h5py.File(sol_fname_h5, "r") as h5f:
        # print("Available datasets in HDF5 file:", list(h5f.keys())) # Use to see datasets in the h5 file
        # print("Datasets inside 'Function':", list(group.keys()))  # Check available datasets
        data = h5f['Function']['Velocity']
        u.x.array[:] = data

        return[u, V, mesh]

def velocity_interpolator_3D(x_grid, y_grid, z_grid, u, v, w):
    # Create interpolators for the 3D velocity field
    u_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), u, bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), v, bounds_error=False, fill_value=None)
    w_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), w, bounds_error=False, fill_value=None)

    def velocity_field(t, pos):
        x, y, z = pos
        u_val = u_interp((x, y, z))
        v_val = v_interp((x, y, z))
        w_val = w_interp((x, y, z))
        return [u_val, v_val, w_val]

    return velocity_field

def stream_trace_3D(coord_grid, vel_grid, seed_points, t_span=(0, 10), max_step=0.1):
    # Compute 3D streamlines starting from given seed points
    xcoords = coord_grid[:, 0]
    ycoords = coord_grid[:, 1]
    zcoords = coord_grid[:, 2]
    vel_grid = vel_grid.x.array.reshape(-1, 3)
    ux = vel_grid[:, 0]  # x-component
    uy = vel_grid[:, 1]  # y-component
    uz = vel_grid[:, 2]  # z-component

    xcoords, ycoords, zcoords, ux, uy, uz = sort_and_reshape(xcoords, ycoords, zcoords, ux, uy, uz)

    velocity_func = velocity_interpolator_3D(xcoords, ycoords, zcoords, ux, uy, uz)
    streamlines = []

    for seed in seed_points:
        sol = solve_ivp(velocity_func, t_span, seed, method='RK45', max_step=max_step, dense_output=True)
        streamlines.append(sol.sol)

    return streamlines

def update_contour(contour):
    gray_img = load_image(img_fname)
    img_contours = get_contours(gray_img)
    contour, mesh_lc = optimize_contour(img_contours[1])
    contour = contour*0.99
    zeros_col = np.zeros((contour.shape[0], 1))
    new_arr = np.hstack((contour, zeros_col))
    return new_arr

def sort_and_reshape(x, y, z, u, v, w):
    """ Sort the grid points and reshape velocity fields to match sorted grids """
    # Sort grid points
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    z_sorted = np.sort(z)

    # Ensure that the velocity fields match the sorted grid order
    # Using np.argsort to get the indices to reorder the velocity field
    x_sorted_indices = np.argsort(x)
    y_sorted_indices = np.argsort(y)
    z_sorted_indices = np.argsort(z)

    # Reorder velocity field arrays
    u_sorted = u[x_sorted_indices]
    u_sorted = u_sorted[:, y_sorted_indices]
    u_sorted = u_sorted[:, :, z_sorted_indices]

    v_sorted = v[x_sorted_indices, :, :]
    v_sorted = v_sorted[:, y_sorted_indices, :]
    v_sorted = v_sorted[:, :, z_sorted_indices]

    w_sorted = w[x_sorted_indices, :, :]
    w_sorted = w_sorted[:, y_sorted_indices, :]
    w_sorted = w_sorted[:, :, z_sorted_indices]

    return x_sorted, y_sorted, z_sorted, u_sorted, v_sorted, w_sorted

print('Reading solution')
u, V, msh = read_sol_file(comm, solname, solname_h5)
gray_img = load_image(img_fname)
img_contours = get_contours(gray_img)
contour, mesh_lc = optimize_contour(img_contours[1])
contour = contour*0.99
contour = update_contour(contour)
msh_coords = msh.geometry.x
# np.set_printoptions(threshold=sys.maxsize)
print(msh_coords.shape)
print(u.x.array)
# u = u.x.array.reshape(-1, 3)
# streamlines = stream_trace_3D(msh_coords, u, contour, t_span=(0, 10), max_step=0.01)