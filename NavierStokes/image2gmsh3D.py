#!/usr/bin/env python

import numpy as np
from skimage import measure
from skimage import io
import skimage as sk
import math
import sys
import os
import time
from rdp import rdp
os.environ["GMSH_FORCE_QUIET"] = "1"
import gmsh

from matplotlib import pyplot as plt

# from image2gmsh_opt2 import get_contours

theta_thresh = 10.0

def get_contours(gray_img):
    height, width = gray_img.shape    
    # Normalize and flip (for some reason)
    contours = sk.measure.find_contours(gray_img, 0.5)

    '''
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(gray_img)
    ax[1].plot(contours[0][:,1], contours[0][:,0])
    ax[2].plot(contours[1][:,1], contours[1][:,0])
    plt.show()
    '''
    
    # print('Found {} contours'.format(len(contours)))
    for n, contour in enumerate(contours):
        # cont_height = max(contour[:,1]) - min(contour[:,1])
        # cont_width = max(contour[:,0]) - min(contour[:,0])

        # if (len(contours) == 1):
            # height = cont_height
            # width = cont_width

        contour[:,1] -= 0.5 * height
        contour[:,1] /= height

        contour[:,0] -= 0.5 * width
        contour[:,0] /= width
        contour[:,0] *= -1.0

    # print("{:d} Contours detected".format(len(contours)))
    # if len(contours) < 2:
    #     print("ERROR: {} contours detected.".format(len(contours)))
    #     print("There should be at least 2 contours")
    #     return 0

    return contours

def check_duplicate_point(pt1, pt2, eps):
    '''
    Instead of comparing absolute distance,
    check if points are co-linear, then check
    orthogonal distance.  This just works 
    better.
    '''
    if (pt1[0] == pt2[0]):
        if math.sqrt((pt1[1]-pt2[1])**2) < eps:
            return True
    elif (pt1[1] == pt2[1]):
        if math.sqrt((pt1[0]-pt2[0])**2) < eps:
            return True
    else:
        return False

def remove_duplicate_points(contour):
    # Figure out a reasonable radius
    max_x = max(contour[:,1])
    min_x = min(contour[:,1])
    
    max_y = max(contour[:,0])
    min_y = min(contour[:,0])
    
    # Set characteristic lengths, epsilon cutoff
    lc = min((max_x - min_x), (max_y - min_y))
    eps = 0.001 * lc
    mesh_lc = 0.01 * lc    
    
    # print('lc = {:.2e}'.format(lc))
    # print('eps = {:.2e}'.format(eps))
    # print('mesh_lc = {:.2e}'.format(mesh_lc))    
    
    duplicate_points = []
    for idx in range(len(contour)):
        point = contour[idx]

        for idx_1 in range(len(contour)):
            if (idx_1 == idx):
                continue
            point_compare = contour[idx_1]
            if (check_duplicate_point(point, point_compare, eps)):
                duplicate_points.append(idx_1)

    # print('Removed {:d} duplicates'.format(len(duplicate_points)))
    contour_out = np.delete(contour, duplicate_points, 0)
    
    return [contour_out, mesh_lc]


def optimize_contour(contour):
    # print("Optimizing contour.")
    dir_flag = 0
    dir_bank = []

    contour_keep = []

    ## Try fft to smooth out 
    x = contour[:,1]
    y = contour[:,0]

    #print(y)
    signal = x + 1j*y
    #print(signal)

    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1])
    cutoff = 0.15
    fft[np.abs(freq) > cutoff] = 0 

    signal_filt = np.fft.ifft(fft)

    contour[:,1] = signal_filt.real
    contour[:,0] = signal_filt.imag
    
    contour = rdp(contour, epsilon=0.0005)

    contour = np.delete(contour, len(contour)-1, 0)

    # cutoff of 0.15, eps of 0.005 works for inner flow
    # Figure out a reasonable radius
    max_x = max(contour[:,1])
    min_x = min(contour[:,1])
    
    max_y = max(contour[:,0])
    min_y = min(contour[:,0])
    
    # Set characteristic lengths, epsilon cutoff
    lc = min((max_x - min_x), (max_y - min_y))
    eps = 0.001 * lc
    mesh_lc = 0.01 * lc    
    
    # print('lc = {:.2e}'.format(lc))
    # print('eps = {:.2e}'.format(eps))
    # print('mesh_lc = {:.2e}'.format(mesh_lc))    

    #contour = reverse_opt_pass(contour)
    return [contour, mesh_lc]


def write_contour_to_file(contour, fname):
    with open(fname, 'w') as f:
        for point in contour:
            f.write('{:.6e}, {:.6e}\n'.format(point[1], point[0]))


def gmsh_3D_extrusion_to_gmsh(mesh_lc, inner_contour, outer_contour, x_outlet, x_extrude, p_idx=1, l_idx=1, loop_idx=1, surf_idx=1):

    gmsh.initialize()
    gmsh.model.add("gmsh_3D_flow")
    g = gmsh.model.occ
    gmsh.option.setNumber("Geometry.OCCAutoFix", 0) # Needed to remove redundant surface, not sure what's going on

    line_init = l_idx

    # print('p_init = {:d}'.format(p_idx))
    # print('\nNumber of points in inner contour: {}'.format(len(inner_contour)))
    # print('Number of points in outer contour: {}'.format(len(outer_contour)))

    # f.write('SetFactory("OpenCASCADE");\n')
    # f.write('Geometry.OCCAutoFix = 0;\n')
    # f.write('lc = {:.3f};\n'.format(1.5*0.015))
    # f.write('lc_0 = {:.3f};\n'.format(1.5*0.015))
    # f.write('lc_1 = {:.3f};\n'.format(1.5*0.01))
    # f.write('lc_outlet = {:.3f};\n'.format(1.5*0.04))
    # f.write('lc_boundary = {:.3f};\n'.format(1.5*0.015))
    # f.write('Mesh.Algorithm = 6;\n\n')

    inlet_inner_surfaces = []
    inlet_outer_surfaces = []
    outlet_surfaces = []
    wall_surfaces = []

    x_inlet = 0.0
    x_outlet = 4.0
    x_extrude = 0.5
    lc = mesh_lc
    lc_1 = lc/2
    lc_outlet = 2*lc

    ############# CREATE OUTER BOX #############

    p_start = p_idx

    # Inlet outer edges
    g.addPoint(0.0, -0.5, -0.5, lc, p_idx)
    g.addPoint(0.0, 0.5, -0.5, lc, p_idx+1)
    g.addPoint(0.0, 0.5, 0.5, lc, p_idx+2)
    g.addPoint(0.0, -0.5, 0.5, lc, p_idx+3)

    g.addLine(p_idx, p_idx+1, l_idx)
    g.addLine(p_idx+1, p_idx+2, l_idx+1)
    g.addLine(p_idx+2, p_idx+3, l_idx+2)
    g.addLine(p_idx+3, p_idx, l_idx+3)

    p_idx += 4
    l_idx += 4

    p_idx_closure = p_idx + 1

    # Outlet 
    g.addPoint(x_outlet, -0.5, -0.5, lc, p_idx)
    g.addPoint(x_outlet, 0.5, -0.5, lc, p_idx+1)
    g.addPoint(x_outlet, 0.5, 0.5, lc, p_idx+2)
    g.addPoint(x_outlet, -0.5, 0.5, lc, p_idx+3)

    g.addLine(p_idx, p_idx+1, l_idx)
    g.addLine(p_idx+1, p_idx+2, l_idx+1)
    g.addLine(p_idx+2, p_idx+3, l_idx+2)
    g.addLine(p_idx+3, p_idx, l_idx+3)
    g.addCurveLoop(list(range(l_idx, l_idx+4)), loop_idx)

    p_idx += 4
    l_idx += 4

    # Outlet
    g.addPlaneSurface([loop_idx],surf_idx)
    outlet_surfaces.append(surf_idx)
    loop_idx += 1
    surf_idx += 1

    # Create walls
    # print(f'l_idx = {l_idx}')
    for i in range(4):
        g.addLine(p_start+i, p_start+i+4, l_idx)
        l_idx += 1

    for i in range(1,4):
        # print(f'iter {i}')
        line_loop = [i, i+9, i+4, i+8]
        # print(line_loop)
        g.addCurveLoop([i, i+8, i+4, i+9], loop_idx)
        # print(f'loop_idx = {loop_idx}')
        # print(f'surf_idx = {surf_idx}')
        g.addPlaneSurface([loop_idx], surf_idx)
        wall_surfaces.append(surf_idx)
        loop_idx += 2
        surf_idx += 1

    l = 4
    g.addCurveLoop([l, l+5, l+4, l+8], loop_idx)
    g.addPlaneSurface([loop_idx], surf_idx)
    wall_surfaces.append(surf_idx)
    loop_idx += 1
    surf_idx += 1

    # ############# END CREATE OUTER BOX #############

    # Inlet inner face, inner contour
    p_idx_closure = p_idx
    p_idx_loop = p_idx
    inlet_inner_contour_points = []
    for point in inner_contour:
        g.addPoint(x_inlet, point[1], point[0], lc, p_idx)
        inlet_inner_contour_points.append(p_idx)
        p_idx += 1

    inlet_inner_contour_lines = []
    for line in range(len(inner_contour)-1):
        g.addLine(p_idx_loop, p_idx_loop+1, l_idx)
        inlet_inner_contour_lines.append(l_idx)
        p_idx_loop += 1
        l_idx += 1
    g.addLine(p_idx_loop, p_idx_closure, l_idx)
    inlet_inner_contour_lines.append(l_idx)
    l_idx += 1

    # Inlet inner face, outer contour
    p_idx_closure = p_idx_loop + 1
    p_idx_loop = p_idx
    inlet_outer_contour_points = []
    for point in outer_contour:
        g.addPoint(x_inlet, point[1], point[0], lc, p_idx)
        inlet_outer_contour_points.append(p_idx)
        p_idx += 1

    inlet_outer_contour_lines = []
    for line in range(len(outer_contour)-1):
        g.addLine(p_idx_loop, p_idx_loop+1, l_idx)
        inlet_outer_contour_lines.append(l_idx)
        p_idx_loop += 1
        l_idx += 1
    g.addLine(p_idx_loop, p_idx_closure, l_idx)
    inlet_outer_contour_lines.append(l_idx)
    l_idx += 1

    # EXTRUDED FACE
    # Extrude inner face, inner contour
    p_idx_closure = p_idx
    p_idx_loop = p_idx
    extrude_inner_contour_points = []
    for point in inner_contour:
        g.addPoint(x_extrude, point[1], point[0], lc, p_idx)
        extrude_inner_contour_points.append(p_idx)
        p_idx += 1

    extrude_inner_contour_lines = []
    for line in range(len(inner_contour)-1):
        g.addLine(p_idx_loop, p_idx_loop+1, l_idx)
        extrude_inner_contour_lines.append(l_idx)
        p_idx_loop += 1
        l_idx += 1
    g.addLine(p_idx_loop, p_idx_closure, l_idx)
    extrude_inner_contour_lines.append(l_idx)
    l_idx += 1

    # Extrude inner face, outer contour
    p_idx_closure = p_idx_loop + 1
    p_idx_loop = p_idx
    extrude_outer_contour_points = []
    for point in outer_contour:
        g.addPoint(x_extrude, point[1], point[0], lc, p_idx)
        extrude_outer_contour_points.append(p_idx)
        p_idx += 1

    extrude_outer_contour_lines = []
    for line in range(len(outer_contour)-1):
        g.addLine(p_idx_loop, p_idx_loop+1, l_idx)
        extrude_outer_contour_lines.append(l_idx)
        p_idx_loop += 1
        l_idx += 1
    g.addLine(p_idx_loop, p_idx_closure, l_idx)
    extrude_outer_contour_lines.append(l_idx)
    l_idx += 1

    # Inner surface connecting lines
    inner_contour_connecting_lines = []
    for i in range(len(inlet_inner_contour_points)):
        g.addLine(inlet_inner_contour_points[i], extrude_inner_contour_points[i], l_idx)
        inner_contour_connecting_lines.append(l_idx)
        l_idx += 1

    # Outer surface connecting lines
    outer_contour_connecting_lines = []
    for i in range(len(inlet_outer_contour_points)):
        g.addLine(inlet_outer_contour_points[i], extrude_outer_contour_points[i], l_idx)
        outer_contour_connecting_lines.append(l_idx)
        l_idx += 1

    # Inner contour surfaces
    # loop_idx = l_idx + 2
    for i in range(len(inlet_inner_contour_lines)-1):
        # print(f'iter = {i}')
        g.addCurveLoop([inlet_inner_contour_lines[i],
            inner_contour_connecting_lines[i], extrude_inner_contour_lines[i], inner_contour_connecting_lines[i+1]], loop_idx)
        g.addPlaneSurface([loop_idx], surf_idx)
        wall_surfaces.append(surf_idx)
        loop_idx += 2
        surf_idx += 1

    g.addCurveLoop([inlet_inner_contour_lines[-1],
        inner_contour_connecting_lines[0], extrude_inner_contour_lines[-1], inner_contour_connecting_lines[-1]], loop_idx)
    g.addPlaneSurface([loop_idx], surf_idx)
    wall_surfaces.append(surf_idx)
    loop_idx += 2
    surf_idx += 1

    # Outer contour surfaces
    # loop_idx = l_idx + 2
    for i in range(len(inlet_outer_contour_lines)-1):
        # g.addCurveLoop([i, i+8, i+4, i+9], loop_idx)
        g.addCurveLoop([inlet_outer_contour_lines[i],
            outer_contour_connecting_lines[i], extrude_outer_contour_lines[i], outer_contour_connecting_lines[i+1]], loop_idx)
        g.addPlaneSurface([loop_idx], surf_idx)
        wall_surfaces.append(surf_idx)
        loop_idx += 2
        surf_idx += 1

    g.addCurveLoop([inlet_outer_contour_lines[-1],
        outer_contour_connecting_lines[0], extrude_outer_contour_lines[-1], outer_contour_connecting_lines[-1]], loop_idx)
    i = g.addPlaneSurface([loop_idx], surf_idx)
    # print(f'i = {i}')
    wall_surfaces.append(surf_idx)
    loop_idx += 2
    surf_idx += 1

    # Extruded wall face
    # print('creating extruded wall face')
    # line_loop_list = extrude_inner_contour_lines
    g.addCurveLoop(extrude_inner_contour_lines, loop_idx)
    loop_idx += 1 
    g.addCurveLoop(extrude_outer_contour_lines, loop_idx)
    loop_idx += 1 
    i = g.addPlaneSurface([loop_idx-2, loop_idx-1], surf_idx)
    # print(f'i = {i}')
    wall_surfaces.append(surf_idx)
    surf_idx +=1

    g.addCurveLoop(inlet_inner_contour_lines, loop_idx)
    
    # g.addCurveLoop(inlet_outer_contour_lines, loop_idx)
    # loop_idx += 1 
    i = g.addPlaneSurface([loop_idx], surf_idx)
    # print(f'i = {i}')
    inlet_inner_surfaces.append(surf_idx)
    loop_idx += 1 
    surf_idx +=1

    g.addCurveLoop([1,2,3,4], loop_idx)
    
    g.addCurveLoop(inlet_outer_contour_lines, loop_idx+1)
    i = g.addPlaneSurface([loop_idx, loop_idx+1], surf_idx)
    # print(f'i = {i}')
    inlet_outer_surfaces.append(surf_idx)
    loop_idx += 2
    surf_idx +=1

    # NEW STUFF
    all_surfaces = inlet_inner_surfaces + inlet_outer_surfaces + wall_surfaces + outlet_surfaces
    # all_surfaces = wall_surfaces + outlet_surfaces

    sl = g.addSurfaceLoop(all_surfaces)
    v = g.addVolume([sl])

    g.synchronize()

    gmsh.model.addPhysicalGroup(2, inlet_inner_surfaces, name = "inlet_1")
    gmsh.model.addPhysicalGroup(2, inlet_outer_surfaces, name = "inlet_2")
    gmsh.model.addPhysicalGroup(2, outlet_surfaces, name = "outlet")
    gmsh.model.addPhysicalGroup(2, wall_surfaces, name = "wall")

    gmsh.model.addPhysicalGroup(3, [1], name = "fluid")

    f_id = 1
    f_id_list = []

    gmsh.model.mesh.field.add('Box', f_id)
    gmsh.model.mesh.field.setNumber(f_id, "VIn", 0.75*lc)
    gmsh.model.mesh.field.setNumber(f_id, "VOut", lc_outlet)
    gmsh.model.mesh.field.setNumber(f_id, "XMin", -0.1)
    gmsh.model.mesh.field.setNumber(f_id, "XMax", x_extrude - 0.25)
    gmsh.model.mesh.field.setNumber(f_id, "YMin", -0.6)
    gmsh.model.mesh.field.setNumber(f_id, "YMax", 0.6)
    gmsh.model.mesh.field.setNumber(f_id, "ZMin", -0.6)
    gmsh.model.mesh.field.setNumber(f_id, "ZMax", 0.6)
    f_id_list.append(f_id)
    f_id += 1

    gmsh.model.mesh.field.add('Box', f_id)
    gmsh.model.mesh.field.setNumber(f_id, "VIn", lc_1)
    gmsh.model.mesh.field.setNumber(f_id, "VOut", lc_outlet)
    gmsh.model.mesh.field.setNumber(f_id, "XMin", x_extrude + 0.25)
    gmsh.model.mesh.field.setNumber(f_id, "XMax", x_extrude + 0.5)
    gmsh.model.mesh.field.setNumber(f_id, "YMin", -0.6)
    gmsh.model.mesh.field.setNumber(f_id, "YMax", 0.6)
    gmsh.model.mesh.field.setNumber(f_id, "ZMin", -0.6)
    gmsh.model.mesh.field.setNumber(f_id, "ZMax", 0.6)
    f_id_list.append(f_id)
    f_id += 1

    gmsh.model.mesh.field.add('Box', f_id)
    gmsh.model.mesh.field.setNumber(f_id, "VIn", 0.75*lc_1)
    gmsh.model.mesh.field.setNumber(f_id, "VOut", lc_outlet)
    gmsh.model.mesh.field.setNumber(f_id, "XMin", x_extrude - 0.1)
    gmsh.model.mesh.field.setNumber(f_id, "XMax", x_extrude + 0.1)
    gmsh.model.mesh.field.setNumber(f_id, "YMin", -0.6)
    gmsh.model.mesh.field.setNumber(f_id, "YMax", 0.6)
    gmsh.model.mesh.field.setNumber(f_id, "ZMin", -0.6)
    gmsh.model.mesh.field.setNumber(f_id, "ZMax", 0.6)
    f_id_list.append(f_id)
    f_id += 1

    gmsh.model.mesh.field.add('Min', f_id)
    gmsh.model.mesh.field.setNumbers(f_id, "FieldsList", f_id_list)
    gmsh.model.mesh.field.setAsBackgroundMesh(f_id)

    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(3)

    return gmsh.model

def load_image(img_fname):
    # print('Loading image {}'.format(img_fname))
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
    
        
def image2gmsh3D(img_fname, mesh_lc_3D, x_outlet, x_extrude, save=False):
    # print('Running image2gmsh')
    job_name = os.path.basename(img_fname.split('.')[0])

    img = load_image(img_fname)
    
    contours = get_contours(img)

    if (len(contours) == 2):
        # print('Processing 2 contour image')

        [contour_inner, mesh_lc_a] = optimize_contour(contours[1])
        [contour_outer, mesh_lc_b] = optimize_contour(contours[0])

        gmsh_model = gmsh_3D_extrusion_to_gmsh(mesh_lc_3D, contour_inner, contour_outer, x_outlet, x_extrude)

        if save:
                gmsh.write('ChannelMesh.msh')
                gmsh.write("ChannelMesh.geo_unrolled")
                # ^ Python's gmsh API doesn't natively save .geo files, but .geo_unrolled files
                # are essentially the same as a .geo file. Just need to rename the file to 
                # mesh_test.geo, and you can open it in gmsh.


    else:
        print('Incorrect number of contours in input image. Exiting.')
        sys.exit(-1)

    return gmsh_model


def main(img_fname, mesh_lc_3D = 0.25):
    x_outlet = 3.0
    x_extrude = 2.0
    save = True

    gmsh_model = image2gmsh3D(img_fname, mesh_lc_3D, x_outlet, x_extrude, save)
    
    return gmsh_model

if __name__ == '__main__':
    img_fname = sys.argv[1]
    main(img_fname)
