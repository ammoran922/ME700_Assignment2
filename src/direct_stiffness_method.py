# import numpy as np
# from math_utils import local_elastic_stiffness_matrix_3D_beam, transformation_matrix_3D, rotation_matrix_3D

# class Node:
#     def __init__(self, node_id, x, y, z):
#         self.id = node_id
#         self.x = x
#         self.y = y
#         self.z = z

# class BoundaryCondition:
#     def __init__(self, node_id, ux, uy, uz, rx, ry, rz):
#         self.node_id = node_id
#         self.ux = ux
#         self.uy = uy
#         self.uz = uz
#         self.rx = rx
#         self.ry = ry
#         self.rz = rz

# class Force:
#     def __init__(self, node_id, fx=0.0, fy=0.0, fz=0.0, mx=0.0, my=0.0, mz=0.0):
#         self.node_id = node_id
#         self.fx = fx
#         self.fy = fy
#         self.fz = fz
#         self.mx = mx
#         self.my = my
#         self.mz = mz

# class Element:
#     def __init__(self, element_id, node_start, node_end, E, nu, A, I_y, I_z, I_rho, J):
#         self.id = element_id
#         self.node_start = node_start
#         self.node_end = node_end
#         self.E = E
#         self.nu = nu
#         self.A = A
#         self.I_y = I_y
#         self.I_z = I_z
#         self.I_rho = I_rho
#         self.J = J

#     def length(self, nodes):
#         s, e = nodes[self.node_start], nodes[self.node_end]
#         return np.sqrt((e.x - s.x) ** 2 + (e.y - s.y) ** 2 + (e.z - s.z) ** 2)

#     def global_stiffness_matrix(self, nodes):
#         L = self.length(nodes)
#         k_local = local_elastic_stiffness_matrix_3D_beam(self.E, self.nu, self.A, L, self.I_y, self.I_z, self.J)
        
#         gamma = rotation_matrix_3D(
#             nodes[self.node_start].x, nodes[self.node_start].y, nodes[self.node_start].z,
#             nodes[self.node_end].x, nodes[self.node_end].y, nodes[self.node_end].z)

#         T = transformation_matrix_3D(gamma)
#         return T.T @ k_local @ T

# class FrameStructure:
#     def __init__(self):
#         self.nodes, self.elements, self.forces, self.boundary_conditions = [], [], [], []

#     def add_node(self, node):
#         self.nodes.append(node)

#     def add_element(self, element):
#         self.elements.append(element)

#     def add_boundary_condition(self, bc):
#         self.boundary_conditions.append(bc)

#     def add_force(self, force):
#         self.forces.append(force)

#     def apply_boundary_conditions(self, K, F):
#         fixed_indices = []
#         for bc in self.boundary_conditions:
#             dof_start = bc.node_id * 6
#             for i, fixed in enumerate([bc.ux, bc.uy, bc.uz, bc.rx, bc.ry, bc.rz]):
#                 if fixed:
#                     fixed_indices.append(dof_start + i)
#         K = np.delete(K, fixed_indices, axis=0)
#         K = np.delete(K, fixed_indices, axis=1)
#         F = np.delete(F, fixed_indices)
#         return K, F, fixed_indices

#     def solve_displacements(self):
#         total_dofs = len(self.nodes) * 6
#         K = np.zeros((total_dofs, total_dofs))
#         F = np.zeros(total_dofs)

#         for e in self.elements:
#             k_global = e.global_stiffness_matrix(self.nodes)
#             idx = list(range(e.node_start * 6, e.node_start * 6 + 6)) + list(range(e.node_end * 6, e.node_end * 6 + 6))
#             for i in range(12):
#                 for j in range(12):
#                     K[idx[i], idx[j]] += k_global[i, j]

#         for l in self.forces:
#             F[l.node_id * 6:l.node_id * 6 + 6] = [l.fx, l.fy, l.fz, l.mx, l.my, l.mz]

#         K_bc, F_bc, fixed_indices = self.apply_boundary_conditions(K, F)
#         U_reduced = np.linalg.solve(K_bc, F_bc)
#         U = np.zeros(total_dofs)
#         free_dofs = np.setdiff1d(np.arange(total_dofs), fixed_indices)
#         U[free_dofs] = U_reduced

#         print("Displacements:", U)

#         # Reaction Forces and Moments
#         reactions = K @ U - F
#         print("Reaction Forces and Moments:", reactions)

#         return U, reactions

import sys
sys.path.append('../src')
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math_utils as mut
from typing import List, Dict, Tuple

class Mesh:
    def __init__(self, nodes: np.ndarray, elements: np.ndarray):
        """
        Initialize the mesh with nodes and elements.
        
        :param nodes: NumPy array of shape (n_nodes, 3) representing node coordinates.
        :param elements: NumPy array of shape (n_elements, 2) representing element connectivity.
        """
        self.nodes = np.array(nodes)
        self.elements = np.array(elements)
        self._validate_inputs()

    def _validate_inputs(self):
        """Validate the inputs for nodes and elements."""
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("Nodes must be a 2D array with shape (n_nodes, 3).")
        if self.elements.ndim != 2 or self.elements.shape[1] != 2:
            raise ValueError("Elements must be a 2D array with shape (n_elements, 2).")
        
        # Validate that all node IDs in elements exist in the nodes array
        max_node_id = len(self.nodes) - 1
        for element in self.elements:
            for node_id in element:
                if node_id < 0 or node_id > max_node_id:
                    raise ValueError(f"Node ID {node_id} in elements does not exist in the nodes array.")
        

class MaterialParams:
    def __init__(self, subdomain_dict: Dict[int, List[float]]):
        """
        Initialize material properties with a dictionary of subdomains and their properties.
        
        :param subdomain_dict: Dictionary where keys are subdomain IDs and values are lists of material properties.
                               Example: {1: [E, nu, A, I_z, I_y, I_p, J, local_z_axis]}
        """
        self.materials = {}
        self.subdomains = {}
        self._initialize_materials(subdomain_dict)

    def _initialize_materials(self, subdomain_dict: Dict[int, List[float]]):
        """Add material properties for each subdomain."""
        for subdomain_id, mat_params in subdomain_dict.items():
            self.add_material(subdomain_id, *mat_params)

    def add_material(self, subdomain_id: int, E: float, nu: float, A: float, I_z: float, I_y: float, I_p: float, J: float, local_z_axis=None):
        """
        Add material properties for a subdomain.
        
        :param subdomain_id: ID of the subdomain.
        :param E: Young's modulus.
        :param nu: Poisson's ratio.
        :param A: Cross-sectional area.
        :param I_z: Moment of inertia about the z-axis.
        :param I_y: Moment of inertia about the y-axis.
        :param I_p: Polar moment of inertia.
        :param J: Torsional constant.
        :param local_z_axis: Local z-axis orientation (optional).
        """
        self._validate_material_properties(E, nu, A, I_z, I_y, I_p, J)
        self.materials[subdomain_id] = {
            'E': E, 'nu': nu, 'A': A, 'I_z': I_z, 'I_y': I_y, 'I_p': I_p, 'J': J, 'local_z_axis': local_z_axis
        }

    def _validate_material_properties(self, E: float, nu: float, A: float, I_z: float, I_y: float, I_p: float, J: float):
        """
        Validate material properties.
        
        :param E: Young's modulus.
        :param nu: Poisson's ratio.
        :param A: Cross-sectional area.
        :param I_z: Moment of inertia about the z-axis.
        :param I_y: Moment of inertia about the y-axis.
        :param I_p: Polar moment of inertia.
        :param J: Torsional constant.
        """
        if E <= 0:
            raise ValueError("Young's modulus (E) must be positive.")
        if not (-1 <= nu <= 0.5):
            raise ValueError("Poisson's ratio (nu) must be between -1 and 0.5.")
        if A <= 0:
            raise ValueError("Cross-sectional area (A) must be positive.")
        if I_z <= 0:
            raise ValueError("Moment of inertia about the z-axis (I_z) must be positive.")
        if I_y <= 0:
            raise ValueError("Moment of inertia about the y-axis (I_y) must be positive.")
        if I_p <= 0:
            raise ValueError("Polar moment of inertia (I_p) must be positive.")
        if J <= 0:
            raise ValueError("Torsional constant (J) must be positive.")

    def assign_subdomain(self, subdomain_id: int, element_ids: List[int]):
        """
        Assign elements to a subdomain.
        
        :param subdomain_id: ID of the subdomain.
        :param element_ids: List of element IDs to assign to the subdomain.
        """
        if subdomain_id not in self.materials:
            raise ValueError(f"Subdomain {subdomain_id} does not exist.")
        self.subdomains[subdomain_id] = element_ids


class BoundaryConditions:
    def __init__(self, supports: Dict[int, Tuple[float, float, float, float, float, float]]):
        """
        Initialize boundary conditions with a dictionary of supports.
        
        :param supports: Dictionary where keys are node IDs and values are tuples of support conditions.
                        Example: {node_id: (u_x, u_y, u_z, theta_x, theta_y, theta_z)}
        """
        self.loads = {}
        self.supports = supports
        self._validate_supports()

    def _validate_supports(self):
        """
        Validate the supports dictionary.
        """
        for node_id, support_conditions in self.supports.items():
            if not isinstance(node_id, int) or node_id < 0:
                raise ValueError(f"Node ID {node_id} must be a non-negative integer.")
            if len(support_conditions) != 6:
                raise ValueError(f"Support conditions for node {node_id} must be a tuple of length 6.")
            for condition in support_conditions:
                if condition is not None and not isinstance(condition, (float, int)):
                    raise ValueError(f"Support condition {condition} for node {node_id} must be a float, int, or None.")

    def _validate_loads(self):
        """
        Validate the loads dictionary.
        """
        for node_id, load_values in self.loads.items():
            if not isinstance(node_id, int) or node_id < 0:
                raise ValueError(f"Node ID {node_id} must be a non-negative integer.")
            if len(load_values) != 6:
                raise ValueError(f"Load values for node {node_id} must be a tuple of length 6.")
            for value in load_values:
                if not isinstance(value, (float, int)):
                    raise ValueError(f"Load value {value} for node {node_id} must be a float or int.")

    def add_load(self, loads: Dict[int, Tuple[float, float, float, float, float, float]]):
        """
        Add a nodal load.
        
        :param loads: Dictionary where keys are node IDs and values are tuples of load values.
                     Example: {node_id: (Fx, Fy, Fz, Mx, My, Mz)}
        """
        self.loads = loads
        self._validate_loads()

    def add_fixed_support(self, node_id: int):
        """Add a fixed support at the specified node."""
        self.supports[node_id] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._validate_supports()

    def add_pinned_support(self, node_id: int):
        """Add a pinned support at the specified node."""
        self.supports[node_id] = (0, 0, 0, None, None, None)
        self._validate_supports()

        
class Solver:
    def __init__(
        self,
        mesh: Mesh,
        material_params: MaterialParams,
        boundary_conditions: BoundaryConditions):
        """
        Initialize the solver with mesh, material properties, and boundary conditions.
        
        :param mesh: Mesh object containing nodes and elements.
        :param material_params: MaterialParams object containing material properties and subdomains.
        :param boundary_conditions: BoundaryConditions object containing supports and loads.
        """
        self.mesh = mesh
        self.material_params = material_params
        self.boundary_conditions = boundary_conditions

    def assemble_global_stiffness_matrix(self) -> np.ndarray:
        """Assemble the global stiffness matrix."""
        n_nodes = len(self.mesh.nodes)
        n_dofs = n_nodes * 6  # 6 DOFs per node
        K_global = np.zeros((n_dofs, n_dofs))

        for element_id, (node1, node2) in enumerate(self.mesh.elements):
            subdomain_id = next((k for k, v in self.material_params.subdomains.items() if element_id in v), None)
            if subdomain_id is None:
                raise ValueError(f"Element {element_id} is not assigned to any subdomain.")
            props = self.material_params.materials[subdomain_id]

            L = np.linalg.norm(self.mesh.nodes[node2] - self.mesh.nodes[node1])
            k_local = mut.local_elastic_stiffness_matrix_3D_beam(props['E'], props['nu'], props['A'], L, props['I_y'], props['I_z'], props['J'])

            node1_loc = self.mesh.nodes[node1]
            node2_loc = self.mesh.nodes[node2]
            gamma = mut.rotation_matrix_3D(node1_loc[0], node1_loc[1], node1_loc[2], node2_loc[0], node2_loc[1], node2_loc[2], props['local_z_axis'])
            T = mut.transformation_matrix_3D(gamma)

            k_global = T.T @ k_local @ T

            dofs = np.array([node1*6, node1*6+1, node1*6+2, node1*6+3, node1*6+4, node1*6+5,
                             node2*6, node2*6+1, node2*6+2, node2*6+3, node2*6+4, node2*6+5])
            for i in range(12):
                for j in range(12):
                    K_global[dofs[i], dofs[j]] += k_global[i, j]

        return K_global

    def solve_system(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the system for unknown displacements and reaction forces."""
        K_global = self.assemble_global_stiffness_matrix()
        n_dofs = K_global.shape[0]
        displacements = np.zeros(n_dofs)
        reactions = np.zeros(n_dofs)

        known_dofs = []
        unknown_dofs = []

        for node_id, (u_x, u_y, u_z, theta_x, theta_y, theta_z) in self.boundary_conditions.supports.items():
            dofs = [node_id * 6 + i for i in range(6)]
            if u_x is not None:
                known_dofs.append(dofs[0])
                displacements[dofs[0]] = u_x
            if u_y is not None:
                known_dofs.append(dofs[1])
                displacements[dofs[1]] = u_y
            if u_z is not None:
                known_dofs.append(dofs[2])
                displacements[dofs[2]] = u_z
            if theta_x is not None:
                known_dofs.append(dofs[3])
                displacements[dofs[3]] = theta_x
            if theta_y is not None:
                known_dofs.append(dofs[4])
                displacements[dofs[4]] = theta_y
            if theta_z is not None:
                known_dofs.append(dofs[5])
                displacements[dofs[5]] = theta_z

        unknown_dofs = [i for i in range(n_dofs) if i not in known_dofs]

        K_uu = K_global[np.ix_(unknown_dofs, unknown_dofs)]
        K_uk = K_global[np.ix_(unknown_dofs, known_dofs)]
        K_ku = K_global[np.ix_(known_dofs, unknown_dofs)]
        K_kk = K_global[np.ix_(known_dofs, known_dofs)]

        F = np.zeros(n_dofs)
        for node_id, (Fx, Fy, Fz, Mx, My, Mz) in self.boundary_conditions.loads.items():
            dofs = [node_id * 6 + i for i in range(6)]
            F[dofs[0]] = Fx
            F[dofs[1]] = Fy
            F[dofs[2]] = Fz
            F[dofs[3]] = Mx
            F[dofs[4]] = My
            F[dofs[5]] = Mz

        F_u = F[unknown_dofs]
        F_k = F[known_dofs]

        displacements[unknown_dofs] = np.linalg.solve(K_uu, F_u)
        reactions[known_dofs] = K_ku @ displacements[unknown_dofs] - F_k

        n_nodes = int(n_dofs / 6)
        displacements = displacements.reshape(n_nodes, 6)
        reactions = reactions.reshape(n_nodes, 6)

        return displacements, reactions

    def generate_mesh_and_solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh and solve the system."""
        displacements, reactions = self.solve_system()

        for node_id, (disp, rxn) in enumerate(zip(displacements, reactions)):
            u, v, w, theta_x, theta_y, theta_z = np.around(disp, decimals=5)
            Fx, Fy, Fz, Mx, My, Mz = np.around(rxn, decimals=5)
            print(f'node {node_id} disp: [u:{u}, v:{v}, w:{w}, theta_x:{theta_x}, theta_y:{theta_y}, theta_z:{theta_z}]')
            print(f'node {node_id} rxn: [Fx:{Fx}, Fy:{Fy}, Fz:{Fz}, Mx:{Mx}, My:{My}, Mz:{Mz}]')
            print('------------------------------------------------------------------')

        return displacements, reactions

    def compute_local_int_forces_and_moments(self,disp:np.ndarray)->np.ndarray:
        disp = disp.flatten()
        element_int_forces = {}
        for element_id, (node1_ind, node2_ind) in enumerate(self.mesh.elements):
            node1_loc = self.mesh.nodes[node1_ind]
            node2_loc = self.mesh.nodes[node2_ind]

            subdomain_id = next((k for k, v in self.material_params.subdomains.items() if element_id in v), None)
            props = self.material_params.materials[subdomain_id]

            L = np.linalg.norm(node1_loc - node2_loc)
            k_local = mut.local_elastic_stiffness_matrix_3D_beam(props['E'], props['nu'], props['A'], L, props['I_y'], props['I_z'], props['J'])

            gamma = mut.rotation_matrix_3D(node1_loc[0], node1_loc[1], node1_loc[2], node2_loc[0], node2_loc[1], node2_loc[2], props['local_z_axis'])
            T = mut.transformation_matrix_3D(gamma)

            dofs = np.array([node1_ind*6, node1_ind*6+1, node1_ind*6+2, node1_ind*6+3, node1_ind*6+4, node1_ind*6+5,
                             node2_ind*6, node2_ind*6+1, node2_ind*6+2, node2_ind*6+3, node2_ind*6+4, node2_ind*6+5])
            
            disp_cur = disp[dofs]
            disp_local = T@disp_cur
            int_forces = k_local@disp_local

            element_int_forces[element_id] = int_forces

        return element_int_forces
    
    def plot_internal_forces(self,element_int_forces:Dict):
        for element_id,int_force in element_int_forces.items():
            node1_ind,node2_ind = self.mesh.elements[element_id]
            node1_loc = self.mesh.nodes[node1_ind]
            node2_loc = self.mesh.nodes[node2_ind]
            local_x = [0,np.linalg.norm(node1_loc - node2_loc)]
            
            fig,ax = plt.subplots(nrows=2,ncols=3,figsize=(15,8))

            ax[0,0].plot(local_x,int_force[[0,6]])
            ax[0,0].set_title('$F_x$')
            ax[0,0].set_xlabel('Distance between nodes')
            ax[0,0].set_ylabel('Force')

            ax[0,1].plot(local_x,int_force[[1,7]])
            ax[0,1].set_title('$F_y$')
            ax[0,1].set_xlabel('Distance between nodes')
            ax[0,1].set_ylabel('Force')

            ax[0,2].plot(local_x,int_force[[2,8]])
            ax[0,2].set_title('$F_z$')
            ax[0,2].set_xlabel('Distance between nodes')
            ax[0,2].set_ylabel('Force')

            ax[1,0].plot(local_x,int_force[[3,9]])
            ax[1,0].set_title('$M_x$')
            ax[1,0].set_xlabel('Distance between nodes')
            ax[1,0].set_ylabel('Moment')

            ax[1,1].plot(local_x,int_force[[4,10]])
            ax[1,1].set_title('$M_y$')
            ax[1,1].set_xlabel('Distance between nodes')
            ax[1,1].set_ylabel('Moment')

            ax[1,2].plot(local_x,int_force[[5,11]])
            ax[1,2].set_title('$M_z$')
            ax[1,2].set_xlabel('Distance between nodes')
            ax[1,2].set_ylabel('Moment')

            fig.suptitle(f'Internal forces and moments for element {element_id}',fontsize=20)
            plt.tight_layout()

    def plot_deformed_structure(self,disp:np.ndarray,scale:float=1.0):
        fig= plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111,projection='3d')

        for element_id,(node1_ind,node2_ind) in enumerate(self.mesh.elements):
            n1_loc = self.mesh.nodes[node1_ind]
            n2_loc = self.mesh.nodes[node2_ind]
            x_ref = [n1_loc[0],n2_loc[0]]
            y_ref = [n1_loc[1],n2_loc[1]]
            z_ref = [n1_loc[2],n2_loc[2]]

            ax.plot(x_ref,y_ref,z_ref,c='b',label='Reference',linewidth=2)

            n1_disp = disp[node1_ind,:]
            n2_disp = disp[node2_ind,:]

            x_cur = [x_ref[0]+scale*n1_disp[0], x_ref[1]+scale*n2_disp[0]]
            y_cur = [y_ref[0]+scale*n1_disp[1], y_ref[1]+scale*n2_disp[1]]
            z_cur = [z_ref[0]+scale*n1_disp[2], z_ref[1]+scale*n2_disp[2]]
            
            ax.plot(x_cur,y_cur,z_cur,c='r',label='Current',linewidth=2)

            if element_id==0:
                ax.legend()
        
        ax.set_title(f'Reference vs Current Configuration w/ scale={scale}',fontsize=20)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    def assemble_global_geometric_stiffness_matrix(self)->np.ndarray:

        n_nodes = len(self.mesh.nodes)
        n_dofs = n_nodes * 6  # 6 DOFs per node
        K_geo_global = np.zeros((n_dofs, n_dofs))

        for element_id,(node1_ind,node2_ind) in enumerate(self.mesh.elements):
            subdomain_id = next((k for k, v in self.material_params.subdomains.items() if element_id in v), None)
            props = self.material_params.materials[subdomain_id]

            n1_loc = self.mesh.nodes[node1_ind]
            n2_loc = self.mesh.nodes[node2_ind]
            L = np.linalg.norm(n1_loc-n2_loc)
            A,Ip = props["A"],props["Ip"]

            cur_int_force = self.internal_forces[element_id]
            Fx1,Fy1,Fz1,Mx1,My1,Mz1,Fx2,Fy2,Fz2,Mx2,My2,Mz2 = cur_int_force

            k_geo=mut.local_geometric_stiffness_matrix_3D_beam(L, A, Ip, Fx2, Mx2, My1, Mz1, My2, Mz2)

            gamma = mut.rotation_matrix_3D(n1_loc[0], n1_loc[1], n1_loc[2], n2_loc[0], n2_loc[1], n2_loc[2], props['local_z_axis'])
            T = mut.transformation_matrix_3D(gamma)

            k_geo_global = T.T @ k_geo @ T

            dofs = np.array([node1_ind*6, node1_ind*6+1, node1_ind*6+2, node1_ind*6+3, node1_ind*6+4, node1_ind*6+5,
                             node2_ind*6, node2_ind*6+1, node2_ind*6+2, node2_ind*6+3, node2_ind*6+4, node2_ind*6+5])

            for i in range(12):
                for j in range(12):
                    K_geo_global[dofs[i], dofs[j]] += k_geo_global[i, j]
            
        return K_geo_global

    def solve_critical_buckling_load(self):
        self.solved_disp,_ = self.solve_system()
        self.internal_forces = self.compute_local_int_forces_and_moments(self.solved_disp)
        
        K_elastic_global = self.assemble_global_stiffness_matrix()
        K_geo_global = self.assemble_global_stiffness_matrix()

        eigvals,eigvecs = sp.linalg.eig(K_elastic_global,K_geo_global)

        real_pos_mask = np.isreal(eigvals) & (eigvals > 0)
        filtered_eigvals = np.real(eigvals[real_pos_mask])
        filtered_eigvecs = eigvecs[:,real_pos_mask]

        sorted_inds = np.argsort(filtered_eigvals)
        filtered_eigvals = filtered_eigvals[sorted_inds]
        filtered_eigvecs = filtered_eigvecs[:,sorted_inds]

        return filtered_eigvals,filtered_eigvecs