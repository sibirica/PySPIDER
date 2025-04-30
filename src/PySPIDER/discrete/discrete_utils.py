from library import *
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

# Notes: this requires observables to already be constructed;
# no power abbreviations
# def construct_from_string(input_str, type_str, obs_dict):
#     # obs_dict: name -> Observable
#     if type_str == "CoarseGrainedPrimitive" or type_str == "CGP":
#         # input_str == "rho[" can happen if method calls itself but is equivalent to length == 1 case
#         token_list = input_str.split('[')
#         if len(token_list) == 1:
#             return CoarseGrainedPrimitive([])
#         else:
#             obs_list = token_list[-1].split(" * ")
#             return CoarseGrainedPrimitive(map(obs_list, lambda x: obs_dict[x]))
#         # makes more sense to construct from the usual constructor
#     elif type_str == "LibraryPrimitive" or type_str == "LP":
#         token_list = input_str.split('[')
#         first_token_list = token_list[0].split()
#         torder = first_token_list.count("dt")
#         xorder = first_token_list.count("dx")
#         return LibraryPrimitive(DerivativeOrder(torder, xorder),
#                                 construct_from_string('rho[' + token_list[-1], "CGP", obs_dict))
#     # elif type_str == "IndexedPrimitive":
#     # not implemented until it seems useful
#     elif type_str == "LibraryTensor":  # most likely not going to be used
#         if input_str == 1:
#             return ConstantTerm()
#         token_list = input_str.split(" @ ")  # it's easier to write the code if @ and * are separate
#         product = construct_from_string(token_list[0], "LibraryPrimitive", obs_dict)
#         for token in token_list[1:]:
#             product *= construct_from_string(token, "LibraryPrimitive", obs_dict)
#     elif type_str == "LibraryTerm" or type_str == "LT":
#         if input_str == 1:
#             return ConstantTerm()
#         token_list = input_str.split(" @ ")
#         obs_list = []
#         index_list = []
#         for token in token_list:
#             obs, inds1, inds2 = term_plus_inds(token, obs_dict)
#             obs_list.append(obs)
#             index_list = index_list + [inds1, inds2]
#         lt = LibraryTensor(obs_list)
#         return LibraryTerm(lt, index_list=index_list)
#     # elif type_str == "IndexedTerm":
#     # not implemented until it seems useful
#     elif type_str == "Equation" or type_str == "EQ":
#         term_list = []
#         coeffs = []
#         token_list = input_str.split(" + ")
#         for token in token_list:
#             if token[0] == '-':
#                 if not token[1].isdigit():
#                     coeff = -1
#                     term = token[1:]
#                 else:
#                     coeff, term = token.split(" * ", 1)
#             elif token[0].isdigit():
#                 coeff, term = token.split(" * ", 1)
#             else:
#                 term = token
#                 coeff = 1
#             coeffs.append(float(coeff))
#             term_list.append(construct_from_string(term, "LibraryTerm", obs_dict).canonicalize())
#         return Equation(term_list, coeffs)
#     else:
#         raise ValueError(type_str + " is not a valid option.")

# def term_plus_inds(string, obs_dict):
#     inds1 = []
#     inds2 = []
#     obs_list = []
#     token_list = string.split('rho')
#     first_token_list = token_list[0].split()
#     obs_token = token_list[-1]
#     if obs_token == "":
#         obs_list = []
#     else:
#         obs_tk_list = obs_token[1:-1].split(" * ")  # remove "[" and "]" parts, get indexed observables
#         for cgp in obs_tk_list:
#             parts = cgp.split("_")
#             obs_list.append(obs_dict[parts[0]])
#             for char in parts[1]:
#                 inds2.append(let_to_num_dict[char])
#     torder = 0
#     xorder = 0
#     for token in first_token_list:
#         if token == "dt":
#             torder += 1
#         else:
#             xorder += 1
#             inds1.append(let_to_num_dict[token[-1]])
#     cgp = CoarseGrainedPrimitive(obs_list)
#     obs = LibraryPrimitive(DerivativeOrder(torder, xorder), cgp)
#     return obs, inds1, inds2