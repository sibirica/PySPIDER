# dependency-free LAMMPS dump file reader
import numpy as np
from findiff import FinDiff
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def dump_to_traj(in_file, out_file, num_dimensions, timestep, L=1, vel_file=None):
    # num_dimensions: 2 (ignore z coordinate) or 3
    # timestep: set in LAMMPS input script, make a copy here
    mode = "metadata"  # toggles between metadata (e.g. what timestep are we on?) or data (particles positions)
    line_is_time = False  # only true if we're on the line where the timestep is stated
    dt_set = False
    prev_time = 0

    line_is_natoms = False
    natoms_set = False

    line_is_bounds = False
    spatial_set = False
    dims = []
    offset = np.zeros(shape=(num_dimensions, 1))

    traj_list = []  # numpy array of trajectories - we'll make it step by step
    with open(in_file, 'r') as f_in:
        for line in f_in:
            if "TIMESTEP" in line:
                mode = "metadata"
                line_is_time = True
            elif "NUMBER" in line:
                line_is_natoms = True
            elif "BOUNDS" in line:
                line_is_bounds = True
            elif "id" in line:
                mode = "data"
                traj = np.zeros(shape=(natoms, num_dimensions))
                traj_list.append(traj)
            elif mode == "data":  # set data
                datafields = list(map(float, line.split()))  # split into [atom, id, type, x, y, (z)]
                atom = int(datafields[0]) - 1
                pos = np.array(datafields[2:2 + num_dimensions])
                traj[atom, :] = pos + offset.transpose()
            else:
                # set metadata
                if line_is_time:  # set timestep
                    line_is_time = False
                    if int(line) != prev_time and not dt_set:
                        dt = timestep * (int(line) - prev_time)
                        dt_set = True
                if line_is_natoms:
                    line_is_natoms = False
                    if not natoms_set:
                        natoms = int(line)
                        natoms_set = True
                if line_is_bounds:
                    line_is_bounds = False
                    if not spatial_set:
                        bounds = list(map(float, line.split()))
                        # offset -= bounds[0] # want grid to start at 0, assume spatial symmetry
                        # dims[:num_dimensions] = [int(np.ceil(bounds[1]-bounds[0]))]*num_dimensions
                        dims.append(1)  # I'm not sure LAMMPS does other size boxes anyway
                        spatial_set = True
    trajs = np.dstack(traj_list)  # stack the list of 2d arrays
    # remake bounds & offset
    offset = np.min(trajs)
    trajs -= offset
    dims[0] = np.max(trajs)
    dims.append(len(traj_list)) # last dim is number of time steps
    if vel_file is None:
        # compute velocities by finite differencing
        traj_diff = FinDiff((num_dimensions, dt, 1), acc=4)
        vs = traj_diff(trajs)  # compute velocities by finite differencing
    else:
        v_list = []
        with open(vel_file, 'r') as f_in:
            for line in f_in:
                if "TIMESTEP" in line:
                    mode = "metadata"
                elif "id" in line:
                    mode = "data"
                    v_slice = np.zeros(shape=(natoms, num_dimensions))
                    v_list.append(v_slice)
                elif mode == "data":  # set data
                    datafields = list(map(float, line.split()))  # split into [atom, vx, vy, (vz)]
                    atom = int(datafields[0]) - 1
                    v = np.array(datafields[1:1 + num_dimensions])
                    v_slice[atom, :] = v
        vs = np.dstack(v_list) / (2*L) # everything is off by length scale rescaling (2*?)L
    # save trajectories, velocities, & dt to out_file
    with open(out_file, 'wb') as f_out:
        np.save(f_out, trajs, allow_pickle=True)
        np.save(f_out, vs, allow_pickle=True)
        np.save(f_out, dt, allow_pickle=True)
        np.save(f_out, dims, allow_pickle=True)


def make_video(out_file, vid_file):  # only works for 2D data at the moment
    fig, ax = plt.subplots(figsize=(6, 6))
    with open(out_file, 'rb') as f:
        pos = np.load(f, allow_pickle=True)
        vs = np.load(f, allow_pickle=True)
    qv = ax.quiver(pos[:, 0, -1], pos[:, 1, -1], 1, 0, clim=[-np.pi, np.pi])

    def animate(i):
        if i % 10 == 0:
            print(i)
        qv.set_offsets(pos[:, :, i])
        norms = np.sqrt(vs[:, 0, i] ** 2 + vs[:, 1, i] ** 2)
        qv.set_UVC(vs[:, 0, i] / norms, vs[:, 1, i] / norms, np.angle(vs[:, 0, i] + 1.0j * vs[:, 1, i]))
        return qv,

    anim = FuncAnimation(fig, animate, np.arange(0, pos.shape[-1]), interval=1, blit=True)
    anim.save(vid_file, fps=30, extra_args=['-vcodec', 'libx264'])
