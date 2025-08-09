### interpolation via clamped B-splines

#from scipy.interpolate import splprep, splev
from scipy.interpolate import BSpline, make_interp_spline
import numpy as np
import matplotlib.pyplot as plt

def interp(data, rate, k=3): # interpolate whole data array
    # recall: data is indexed by (particle, dimension, time)
    #u = np.hstack([[0]*k, np.linspace(0, 1, num=data.shape[-1]), [1]*k]) # clamp both ends by repeating extra k times
    u = np.linspace(0, 1, num=data.shape[-1])
    eval_pts = np.linspace(0, 1, num=(data.shape[-1]-1)*rate+1)
    outputs = [
        interp_particle(data[particle, ...], rate, k, u, eval_pts) 
        for particle in range(data.shape[0])
    ]
    fine_data = [output[0] for output in outputs]
    splines = [output[1] for output in outputs]
    fine_data = np.transpose(np.dstack(fine_data))
    #print(fine_data.shape)
    return fine_data, splines

# interpolate individual particle trajectory (or observable time series)
def interp_particle(part_data, rate, k, u, eval_pts):
    #if len(part_data.shape)==1:
        #tck = splprep([part_data], u=u, k=k)
    #else:
        # split up into x, y etc. lists and clamp by duplicating first and last 
        # point an extra k times
        # coords_list = [clamp_k(l[0, :], k) for l in np.split(part_data, part_data.shape[0], axis=0)]
    #    print(coords_list)
        #tck = splprep(coords_list, u=u, k=k)
    spline = make_interp_spline(u, part_data.T, k, bc_type="clamped")
    fine_data = spline(eval_pts)
    #fine_data = splev(eval_pts, tck)
    return fine_data, spline
              
#def clamp_k(vector, k):
#    return np.hstack([[vector[0]]*k, vector, [vector[1]]*k])

def plot_interp(part_data, fine_data): # plot interpolation for one particle
    fig, ax = plt.subplots(1, 1)
    ax.plot(fine_data[0, :], fine_data[1, :], ',')
    ax.plot(part_data[0, :], part_data[1, :], 'go', markersize=4, fillstyle='none')

    plt.show()