# import sys
# sys.path.append('../src')
# import direct_stiffness_method as dsm
# import pytest

# def run_test():
#     E, nu = 1000, 0.3
#     b, h = 0.5, 1.0
#     A = b * h
#     I_y = h * b ** 3 / 12
#     I_z = b * h ** 3 / 12
#     I_rho = b * h / 12 * (b**2 + h**2) 
#     J = 0.02861

#     frame = dsm.FrameStructure()
#     frame.add_node(dsm.Node(0, 0, 0, 10))
#     frame.add_node(dsm.Node(1, 15, 0, 10))
#     frame.add_node(dsm.Node(2, 15, 0, 0))

#     frame.add_element(dsm.Element(0, 0, 1, E, nu, A, I_y, I_z, I_rho, J))
#     frame.add_element(dsm.Element(1, 1, 2, E, nu, A, I_y, I_z, I_rho, J))

#     frame.add_boundary_condition(dsm.BoundaryCondition(0, True, True, True, True, True, True))
#     frame.add_boundary_condition(dsm.BoundaryCondition(2, True, True, True, False, False, False))

#     frame.add_force(dsm.Force(1, fx=0.1, fy=0.05, fz=-0.07, mx=0.05, my=-0.1, mz=0.25))
#     frame.solve_displacements()

# run_test()

import sys
sys.path.append('../src')
import pytest
import numpy as np
from direct_stiffness_method import Mesh, MaterialParams, BoundaryConditions, Solver


@pytest.fixture
def knownKGlobal():
    E,nu,A,Iz,Iy,Ip,J,L=210e9,0.3,0.01,1e-5,1e-5,1e-5,1e-5,1.0
    known = np.zeros((12, 12))
    # Axial terms - extension of local x axis
    axial_stiffness = E * A / L
    known[0, 0] = axial_stiffness
    known[0, 6] = -axial_stiffness
    known[6, 0] = -axial_stiffness
    known[6, 6] = axial_stiffness
    # Torsion terms - rotation about local x axis
    torsional_stiffness = E * J / (2.0 * (1 + nu) * L)
    known[3, 3] = torsional_stiffness
    known[3, 9] = -torsional_stiffness
    known[9, 3] = -torsional_stiffness
    known[9, 9] = torsional_stiffness
    # Bending terms - bending about local z axis
    known[1, 1] = E * 12.0 * Iz / L ** 3.0
    known[1, 7] = E * -12.0 * Iz / L ** 3.0
    known[7, 1] = E * -12.0 * Iz / L ** 3.0
    known[7, 7] = E * 12.0 * Iz / L ** 3.0
    known[1, 5] = E * 6.0 * Iz / L ** 2.0
    known[5, 1] = E * 6.0 * Iz / L ** 2.0
    known[1, 11] = E * 6.0 * Iz / L ** 2.0
    known[11, 1] = E * 6.0 * Iz / L ** 2.0
    known[5, 7] = E * -6.0 * Iz / L ** 2.0
    known[7, 5] = E * -6.0 * Iz / L ** 2.0
    known[7, 11] = E * -6.0 * Iz / L ** 2.0
    known[11, 7] = E * -6.0 * Iz / L ** 2.0
    known[5, 5] = E * 4.0 * Iz / L
    known[11, 11] = E * 4.0 * Iz / L
    known[5, 11] = E * 2.0 * Iz / L
    known[11, 5] = E * 2.0 * Iz / L
    # Bending terms - bending about local y axis
    known[2, 2] = E * 12.0 * Iy / L ** 3.0
    known[2, 8] = E * -12.0 * Iy / L ** 3.0
    known[8, 2] = E * -12.0 * Iy / L ** 3.0
    known[8, 8] = E * 12.0 * Iy / L ** 3.0
    known[2, 4] = E * -6.0 * Iy / L ** 2.0
    known[4, 2] = E * -6.0 * Iy / L ** 2.0
    known[2, 10] = E * -6.0 * Iy / L ** 2.0
    known[10, 2] = E * -6.0 * Iy / L ** 2.0
    known[4, 8] = E * 6.0 * Iy / L ** 2.0
    known[8, 4] = E * 6.0 * Iy / L ** 2.0
    known[8, 10] = E * 6.0 * Iy / L ** 2.0
    known[10, 8] = E * 6.0 * Iy / L ** 2.0
    known[4, 4] = E * 4.0 * Iy / L
    known[10, 10] = E * 4.0 * Iy / L
    known[4, 10] = E * 2.0 * Iy / L
    known[10, 4] = E * 2.0 * Iy / L
    return known


def test_mesh_input_validation():
    # Test valid node IDs in elements
    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    elements = np.array([[0, 1], [1, 2]])  # Valid elements
    mesh = Mesh(nodes, elements)
    assert mesh.nodes.shape == (3, 3)
    assert mesh.elements.shape == (2, 2)

    # Test invalid node IDs in elements
    with pytest.raises(ValueError):
        invalid_elements = np.array([[0, 3], [1, 2]])  # Node ID 3 does not exist
        Mesh(nodes, invalid_elements)

    # Test empty nodes or elements
    with pytest.raises(ValueError):
        Mesh(np.array([]), np.array([[0, 1]]))  # Empty nodes
    with pytest.raises(ValueError):
        Mesh(nodes, np.array([]))  # Empty elements


def test_material_properties_validation():
    # Test valid material properties
    material_params = MaterialParams({1: [210e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, 1e-6]})
    assert material_params.materials[1]['E'] == 210e9

    # Test invalid Young's modulus
    with pytest.raises(ValueError):
        MaterialParams({1: [-210e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, 1e-6]})  # Negative E

    # Test invalid Poisson's ratio
    with pytest.raises(ValueError):
        MaterialParams({1: [210e9, 1.5, 0.01, 1e-6, 1e-6, 1e-6, 1e-6]})  # nu > 0.5

    # Test invalid cross-sectional area
    with pytest.raises(ValueError):
        MaterialParams({1: [210e9, 0.3, -0.01, 1e-6, 1e-6, 1e-6, 1e-6]})  # Negative A

    # Test invalid moments of inertia
    with pytest.raises(ValueError):
        MaterialParams({1: [210e9, 0.3, 0.01, -1e-6, 1e-6, 1e-6, 1e-6]})  # Negative I_z

    # Test invalid torsional constant
    with pytest.raises(ValueError):
        MaterialParams({1: [210e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, -1e-6]})  # Negative J


def test_boundary_conditions_validation():
    # Test valid supports
    supports = {0: (0.0, 0.0, 0.0, None, None, None)}
    bc = BoundaryConditions(supports)
    assert bc.supports[0] == (0.0, 0.0, 0.0, None, None, None)

    # Test invalid node ID in supports
    with pytest.raises(ValueError):
        BoundaryConditions({-1: (0.0, 0.0, 0.0, None, None, None)})  # Negative node ID

    # Test invalid support conditions (wrong length)
    with pytest.raises(ValueError):
        BoundaryConditions({0: (0.0, 0.0, 0.0, None)})  # Tuple length != 6

    # Test invalid support conditions (wrong type)
    with pytest.raises(ValueError):
        BoundaryConditions({0: (0.0, 0.0, 0.0, "invalid", None, None)})  # Non-float condition

    # Test valid loads
    bc.add_load({0: (1000, 0, 0, 0, 0, 0)})
    assert bc.loads[0] == (1000, 0, 0, 0, 0, 0)

    # Test invalid node ID in loads
    with pytest.raises(ValueError):
        bc.add_load({-1: (1000, 0, 0, 0, 0, 0)})  # Negative node ID

    # Test invalid load values (wrong length)
    with pytest.raises(ValueError):
        bc.add_load({0: (1000, 0, 0, 0)})  # Tuple length != 6

    # Test invalid load values (wrong type)
    with pytest.raises(ValueError):
        bc.add_load({0: (1000, 0, 0, "invalid", 0, 0)})  # Non-float value


def test_empty_mesh():
    # Test empty mesh
    with pytest.raises(ValueError):
        Mesh(np.array([]), np.array([]))


def test_invalid_subdomain_assignment():
    nodes = np.array([[0, 0, 0], [1, 0, 0]])
    elements = np.array([[0, 1]])
    material_params = MaterialParams({1: [210e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, 1e-6]})
    mesh = Mesh(nodes, elements)

    # Test invalid subdomain assignment
    with pytest.raises(ValueError):
        solver = Solver(mesh, material_params, BoundaryConditions({}))
        solver.assemble_global_stiffness_matrix()


def test_global_stiffness_matrix():
    # Provide your own matrix for validation
    nodes = np.array([[0, 0, 0], [1, 0, 0]])
    elements = np.array([[0, 1]])
    material_params = MaterialParams({1: [210e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, 1e-6]})
    material_params.assign_subdomain(1, [0])
    mesh = Mesh(nodes, elements)
    bc = BoundaryConditions({0: (0.0, 0.0, 0.0, None, None, None)})
    solver = Solver(mesh, material_params, bc)

    # Assemble global stiffness matrix
    K_global = solver.assemble_global_stiffness_matrix()
    assert K_global.shape == (12, 12)  # 2 nodes * 6 DOFs

def test_assemble_global_stiffness_matrix(knownKGlobal):
    # Provide your own matrix for validation
    nodes = np.array([[0, 0, 0], [1, 0, 0]])
    elements = np.array([[0, 1]])
    material_params = MaterialParams({1: [210e9, 0.3, 0.01, 1e-5, 1e-5, 1e-5, 1e-5]})
    material_params.assign_subdomain(1, [0])
    mesh = Mesh(nodes, elements)
    bc = BoundaryConditions({0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)})
    bc.add_load({1: (0, -1000, 0, 0, 0, 0)})
    solver = Solver(mesh, material_params, bc)

    # Assemble global stiffness matrix
    K_global = solver.assemble_global_stiffness_matrix()

    assert np.allclose(knownKGlobal,K_global)

def test_solve_system():
    # Provide your own example for validation
    nodes = np.array([[0, 0, 0], [1, 0, 0]])
    elements = np.array([[0, 1]])
    material_params = MaterialParams({1: [210e9, 0.3, 0.01, 1e-5, 1e-5, 1e-5, 1e-5,[0,0,1]]})
    material_params.assign_subdomain(1, [0])
    mesh = Mesh(nodes, elements)
    bc = BoundaryConditions({0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)})
    bc.add_load({1: (0, -1000, 0, 0, 0, 0)})
    solver = Solver(mesh, material_params, bc)

    known_disps = [0,-0.00015873,0,0,0,-0.0002381]
    known_rxns = [0,1000,0,0,0,1000]

    # Solve the system
    displacements, reactions = solver.solve_system()
    assert np.allclose(displacements[1,:],known_disps)
    assert np.allclose(reactions[0,:],known_rxns)


def test_compute_local_int_forces_and_moments():
    # Provide your own example for validation
    nodes = np.array([[0, 0, 0], [1, 0, 0]])
    elements = np.array([[0, 1]])
    material_params = MaterialParams({1: [210e9, 0.3, 0.01, 1e-5, 1e-5, 1e-5, 1e-5,[0,0,1]]})
    material_params.assign_subdomain(1, [0])
    mesh = Mesh(nodes, elements)
    bc = BoundaryConditions({0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)})
    bc.add_load({1: (0, -1000, 0, 0, 0, 0)})
    solver = Solver(mesh, material_params, bc)
    displacements, reactions = solver.solve_system()
    element_int_forces = solver.compute_local_int_forces_and_moments(displacements)

    known_int_forces = {0:np.array([0.0,1000,0.0,0.0,0.0,1000,0.0,-1000,0.0,0.0,0.0,0.0])}
    
    assert np.allclose(known_int_forces[0],element_int_forces[0])


def test_plot_internal_forces():
    # Provide your own example for validation
    nodes = np.array([[0, 0, 0], [1, 0, 0]])
    elements = np.array([[0, 1]])
    material_params = MaterialParams({1: [210e9, 0.3, 0.01, 1e-5, 1e-5, 1e-5, 1e-5,[0,0,1]]})
    material_params.assign_subdomain(1, [0])
    mesh = Mesh(nodes, elements)
    bc = BoundaryConditions({0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)})
    bc.add_load({1: (0, -1000, 0, 0, 0, 0)})
    solver = Solver(mesh, material_params, bc)
    displacements, reactions = solver.solve_system()
    element_int_forces = solver.compute_local_int_forces_and_moments(displacements)
    solver.plot_internal_forces(element_int_forces)
    
    assert True


def test_plot_deformed_structure():
    # Provide your own example for validation
    nodes = np.array([[0, 0, 0], [1, 0, 0]])
    elements = np.array([[0, 1]])
    material_params = MaterialParams({1: [210e9, 0.3, 0.01, 1e-5, 1e-5, 1e-5, 1e-5,[0,0,1]]})
    material_params.assign_subdomain(1, [0])
    mesh = Mesh(nodes, elements)
    bc = BoundaryConditions({0: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)})
    bc.add_load({1: (0, -1000, 0, 0, 0, 0)})
    solver = Solver(mesh, material_params, bc)
    displacements, reactions = solver.solve_system()
    solver.plot_deformed_structure(displacements)
    
    assert True


# def test_solve_system_with_geometric_nonlinearities():