#!/usr/bin/env python

import numpy as np
import sys
import os
from mpi4py import MPI

from streamtrace import for_and_rev_streamtrace
from NavierStokesChannelFlow import solve_NS_flow, make_output_folder, write_run_metadata, save_navier_stokes_solution
from dolfinx import geometry

comm = MPI.COMM_WORLD
global bb_tree
global mesh
global uh
global uvw_data
global xyz_data

def save_figs(img_fname, inner_contour_fig, inner_contour_mesh_fig, seeds, final_output, rev_streamtrace_fig, num_seeds):
    print('Saving Figures')
    inner_contour_fig.savefig("inner_contour.svg")
    inner_contour_mesh_fig.savefig("inner_mesh.svg")
    print(img_fname)
    img_fname = os.path.basename(img_fname)
    print(img_fname)
    img_fname = img_fname.removesuffix(".png")
    print(img_fname)
    rev_streamtrace_fig.savefig(f"rev_trace_{img_fname}_{num_seeds}.svg")
    np.savetxt("rev_seeds.csv", seeds, delimiter=",")
    np.savetxt("final_output.csv", final_output, delimiter=",")

def main():
    global bb_tree
    global msh
    global uh
    global uvw_data
    global xyz_data
    num_seeds = 50
    limits = 0.5
    print(os.getcwd())
    msh, uh, uvw_data, xyz_data, Re, img_fname, channel_mesh_size, V, Q, flow_ratio, u, p = solve_NS_flow()
    Folder_name, img_name = make_output_folder(Re, img_fname, channel_mesh_size)
    write_run_metadata(Folder_name, Re, img_fname, flow_ratio, channel_mesh_size, V, Q, img_name)
    save_navier_stokes_solution(u, p, msh, Folder_name, Re)
    print(os.getcwd())
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    rev_streamtrace_fig, inner_contour_fig, inner_contour_mesh_fig, seeds, final_output = for_and_rev_streamtrace(num_seeds, limits, img_fname, msh, u, uvw_data, xyz_data, msh)
    save_figs(img_fname, inner_contour_fig, inner_contour_mesh_fig, seeds, final_output, rev_streamtrace_fig, num_seeds)

if __name__ == "__main__":
    main()