#!/usr/bin/env python

import numpy as np
import sys
import os
from mpi4py import MPI

from streamtrace import for_and_rev_streamtrace
from NavierStokesChannelFlow import solve_NS_flow, make_output_folder, write_run_metadata, save_navier_stokes_solution
from dolfinx import geometry

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
global bb_tree
global mesh
global uh
global uvw_data
global xyz_data

def save_figs(img_fname, inner_contour_fig, inner_contour_mesh_fig, seeds, final_output, rev_streamtrace_fig, num_seeds, folder_name):
    print(folder_name)
    os.chdir(os.path.join(os.getcwd(), folder_name))
    if rank == 0:
        print('Saving Figures', flush=True)
    inner_contour_fig.savefig("inner_contour.svg")
    inner_contour_mesh_fig.savefig("inner_mesh.svg")
    if rank == 0:
        print(img_fname, flush=True)
    img_fname = os.path.basename(img_fname)
    if rank == 0:
        print(img_fname, flush=True)
    img_fname = img_fname.removesuffix(".png")
    if rank == 0:
        print(img_fname, flush=True)
    rev_streamtrace_fig.savefig(f"rev_trace_{img_fname}_{num_seeds}.svg")
    np.savetxt("rev_seeds.csv", seeds, delimiter=",")
    np.savetxt("final_output.csv", final_output, delimiter=",")

def run_trace_save():
    try:
        num_seeds = 400
        limits = 1
        if rank == 0:
            print(f"[Rank {rank}] Starting solve_NS_flow()...", flush=True)
        msh, uh, uvw_data, xyz_data, Re, img_fname, channel_mesh_size, V, Q, flow_ratio, u, p = solve_NS_flow()
        if rank == 0:
            print(f"[Rank {rank}] Finished solve_NS_flow()", flush=True)

        comm.Barrier()

        Folder_name, img_name = make_output_folder(Re, img_fname, channel_mesh_size, comm, rank)

        comm.Barrier()
        if rank == 0:
            print(f"[Rank {rank}] Writing metadata and saving NS solution...", flush=True)
        write_run_metadata(Folder_name, Re, img_fname, flow_ratio, channel_mesh_size, V, Q, img_name, comm, rank)
        save_navier_stokes_solution(u, p, msh, Folder_name, Re, comm, rank)
        comm.Barrier()
        if rank == 0:
            print(f"[Rank {rank}] Starting for_and_rev_streamtrace...", flush=True)
        rev_streamtrace_fig, inner_contour_fig, inner_contour_mesh_fig, seeds, final_output = for_and_rev_streamtrace(
            num_seeds, limits, img_fname, msh, u, uvw_data, xyz_data, Re, Folder_name
        )
        if rank == 0:
            print(f"[Rank {rank}] Streamtrace finished.", flush=True)

        if rank == 0:
            print(f"[Rank {rank}] Saving figures...", flush=True)
            save_figs(img_fname, inner_contour_fig, inner_contour_mesh_fig, seeds, final_output, rev_streamtrace_fig, num_seeds, Folder_name)
        if rank == 0:
            print(f"[Rank {rank}] run_trace_save completed successfully.", flush=True)
        
    except Exception as e:
        print(f"[Rank {rank}] ERROR in run_trace_save: {e}", flush=True)
        import traceback
        traceback.print_exc()


def main():
    run_trace_save()

if __name__ == "__main__":
    main()