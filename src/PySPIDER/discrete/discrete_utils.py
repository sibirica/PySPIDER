import numpy as np

def unroll(traj_array, world_size): # transform trajectory array to eliminate jumps over periodic boundary
    return np.stack([unroll_particle(traj_array[i, :, :], world_size) for i in range(traj_array.shape[0])], axis=0) 
# transform each particle individually
        
def unroll_particle(part_array, world_size):
    copy_array = np.copy(part_array)
    for j in range(part_array.shape[0]):
        for k in range(part_array.shape[1]-1):
            if np.abs(part_array[j, k]-part_array[j, k+1]) > world_size[j]/2: # boundary crossed
                copy_array[j, k+1:] += np.sign(part_array[j, k]-part_array[j, k+1])*world_size[j]
    return copy_array
    
def roll(traj_array, world_size): # inverse transform unrolled trajectory array using modulus
    return np.stack([traj_array[:, j, :] % world_size[j] for j in range(traj_array.shape[1])], axis=1)