import numpy as np
from pathlib import Path
import math_utils as mu

# forces = global * postions
# four 3x3 matrixes to manipulate for solving partion matrix

def partition_nodes(displacement_forces: np.ndarray, positions: np.ndarray, final_global_matrix: np.ndarray):
    
    positions = final_global_matrix * displacement_forces # gives a 12 x 1
    pass

# Solve unknown variables(postions at free DOF and forces at supported DOF)
def solve_unknowns(partitioned_matrix: np.ndarray):
    pass

# Computing the local elements forces and moments
def post_process(solved_matrix: np.ndarray):
    pass

def direct_stiffness_method(displacement_forces: np.ndarray, positions: np.ndarray, gamma_rotation_matrix: np.ndarray, E: float, nu: float, A: float, L: float, Iy: float, Iz: float, J: float):

    local_matrix = mu.local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)

    transformation_matrix = mu.transformation_matrix_3D(gamma_rotation_matrix)
    
    global_matrix = local_matrix * transformation_matrix
    
    final_global_matrix = mu.gamma_rotation_matrix_3D(global_matrix)

    # partitioned_matrix = partition_nodes(displacement_forces: np.ndarray, positions: np.ndarray, final_global_matrix: np.ndarray)
    # solved_matrix = solve_unknowns(partitioned_matrix)
    # post_processed_matrix = post_process(solved_matrix) 
    
    # post_processed_matrix is final answer
