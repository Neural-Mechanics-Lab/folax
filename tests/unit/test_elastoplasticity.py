import pytest
import unittest 
import os
import numpy as np
import jax.numpy as jnp
from fol.loss_functions.mechanical_elastoplasticity import ElastoplasticityLoss2DQuad
from fol.loss_functions.mechanical_elastoplasticity import ElastoplasticityLoss3DTetra
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import *


class TestElastoplasticity(unittest.TestCase):
    """Unit tests for elastoplastic material behavior in 2D and 3D"""

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self, request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def test_tetra(self):
        """Test tetrahedral element computation"""
        
        tet_points_coordinates = jnp.array([
            [0.1, 0.1, 0.1],
            [0.28739360416666665, 0.27808503701741405, 0.05672979583333333],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.1]
        ])

        fe_mesh = Mesh("", ".")
        fe_mesh.node_ids = jnp.arange(len(tet_points_coordinates))
        fe_mesh.nodes_coordinates = tet_points_coordinates
        fe_mesh.elements_nodes = {"tetra": fe_mesh.node_ids.reshape(1, -1)}

        material_dict = {
            "young_modulus": 3.0,
            "poisson_ratio": 0.3,
            "iso_hardening_parameter_1": 0.4,
            "iso_hardening_param_2": 10.0,
            "yield_limit": 0.2
        }

        mechanical_loss_3d = ElastoplasticityLoss3DTetra(
            "mechanical_loss_3d",
            loss_settings={
                "dirichlet_bc_dict": {"Ux": {}, "Uy": {}, "Uz": {}},
                "material_dict": material_dict,
                "body_foce": jnp.array([[1], [2], [3]])
            },
            fe_mesh=fe_mesh
        )
        mechanical_loss_3d.Initialize()
        
        # Rigid body translation 
        displacements = jnp.ones((12, 1))
        
        elem_controls = jnp.array([1.0])
        
        # State: [ε_p_xx, ε_p_yy, ε_p_zz, ε_p_xy, ε_p_yz, ε_p_xz, internal_parameter]
        num_gp = 1
        state_size = 7
        element_state_gps = jnp.zeros((num_gp, state_size))

        en, gps_state, residuals, stiffness = mechanical_loss_3d.ComputeElement(
            tet_points_coordinates,
            elem_controls,
            displacements,
            element_state_gps
        )

        np.testing.assert_allclose(stiffness, jnp.array([
            [ 0.20516104,  0.07489204,  0.        , -0.2404595 , -0.06452059,  0.        ,  0.10404732,  0.02791819,  0.20824468, -0.06874882, -0.03828965, -0.20824468],
            [ 0.07489204,  0.13731398,  0.        , -0.0549052 , -0.14439207,  0.        ,  0.0237576 ,  0.06247874,  0.08290199, -0.04374444, -0.05540066, -0.08290199],
            [ 0.        ,  0.        ,  0.12453636,  0.        ,  0.        , -0.13994603,  0.27765957,  0.11053599,  0.06055493, -0.27765957, -0.11053599, -0.04514527],
            [-0.24045952, -0.0549052 ,  0.        ,  0.2944494 ,  0.03248741,  0.        , -0.12740886, -0.01405737, -0.25961536,  0.07341897,  0.03647517,  0.25961536],
            [-0.06452058, -0.14439207,  0.        ,  0.03248741,  0.17068782,  0.        , -0.01405737, -0.07385697, -0.02884615,  0.04609055,  0.04756121,  0.02884615],
            [ 0.        ,  0.        , -0.13994603,  0.        ,  0.        ,  0.16914082, -0.34615383, -0.03846154, -0.07318757,  0.34615383,  0.03846154,  0.04399279],
            [ 0.10404732,  0.0237576 ,  0.27765957, -0.12740885, -0.01405737, -0.3461538 ,  0.7722944 ,  0.00608265,  0.26211756, -0.74893284, -0.01578288, -0.19362329],
            [ 0.02791819,  0.06247874,  0.11053598, -0.01405737, -0.07385697, -0.03846154,  0.00608265,  0.7491224 ,  0.02912419, -0.01994348, -0.73774415, -0.10119865],
            [ 0.20824468,  0.08290199,  0.06055493, -0.25961536, -0.02884615, -0.07318757,  0.26211756,  0.02912419,  1.286706  , -0.21074685, -0.08318003, -1.2740734 ],
            [-0.06874882, -0.04374444, -0.27765957,  0.07341897,  0.04609055,  0.3461538 , -0.74893284, -0.01994348, -0.21074685,  0.7442627 ,  0.01759737,  0.14225261],
            [-0.03828965, -0.05540066, -0.11053598,  0.03647517,  0.04756121,  0.03846154, -0.01578288, -0.73774415, -0.08318003,  0.01759737,  0.74558365,  0.15525448],
            [-0.20824468, -0.08290199, -0.04514527,  0.25961536,  0.02884615,  0.0439928 , -0.1936233 , -0.10119865, -1.2740734 ,  0.14225261,  0.15525448,  1.2752258 ]
        ]), rtol=1e-5, atol=1e-5)

        np.testing.assert_allclose(residuals.flatten(), jnp.array([
            -0.00077693, -0.00155386, -0.00233078, 
            -0.00077693, -0.00155386, -0.00233078,
            -0.00077693, -0.00155386, -0.00233079, 
            -0.00077693, -0.00155385, -0.00233078
        ]), rtol=1e-5, atol=1e-5)

    def test_quad(self):
        """Test 2D quadrilateral element computation"""
        
        quad_points_coordinates = jnp.array([
            [3.00, 0.00, 0.00],
            [2.00, 0.75, 0.00],
            [0.75, 1.00, 0.00],
            [0.00, 0.00, 0.00]
        ])
        
        fe_mesh = Mesh("", ".")
        fe_mesh.node_ids = jnp.arange(len(quad_points_coordinates))
        fe_mesh.nodes_coordinates = quad_points_coordinates
        fe_mesh.elements_nodes = {"quad": fe_mesh.node_ids.reshape(1, -1)}

        material_dict = {
            "young_modulus": 3.0,
            "poisson_ratio": 0.3,
            "iso_hardening_parameter_1": 0.4,
            "iso_hardening_param_2": 10.0,
            "yield_limit": 0.2
        }

        mechanical_loss_2d = ElastoplasticityLoss2DQuad(
            "mechanical_loss_2d",
            loss_settings={
                "dirichlet_bc_dict": {"Ux": {}, "Uy": {}},
                "material_dict": material_dict,
                "body_foce": jnp.array([[1], [2]])
            },
            fe_mesh=fe_mesh
        )
        mechanical_loss_2d.Initialize()

        # Displacement field (Rigid body motion)
        displacements = jnp.ones((8, 1))

        # Controls: Scalar density
        elem_controls = jnp.array([1.0])
        
        # State: [ε_p_xx, ε_p_yy, ε_p_xy, internal_parameter]
        num_gp = 4 
        state_size = 4  
        element_state_gps = jnp.zeros((num_gp, state_size))

        en, gps_state, residuals, stiffness = mechanical_loss_2d.ComputeElement(
            quad_points_coordinates,
            elem_controls,
            displacements,
            element_state_gps
        )

        np.testing.assert_allclose(stiffness, jnp.array([
            [ 1.9688053 , -0.76231086, -1.851286  , -0.35307044, -0.9192276 ,  0.9138413 ,  0.8017084 ,  0.2015399 ],
            [-0.76231086,  2.969756  , -0.06460892, -3.4887218 ,  0.9138413 , -1.42192   , -0.08692163,  1.9408858 ],
            [-1.851286  , -0.06460901,  3.913721  ,  1.4893513 , -0.33913708, -0.2516288 , -1.7232977 , -1.1731135 ],
            [-0.35307047, -3.4887218 ,  1.4893513 ,  5.7130857 ,  0.03683268,  0.2584547 , -1.1731135 , -2.4828186 ],
            [-0.91922754,  0.9138413 , -0.33913708,  0.03683267,  2.5620449 , -0.67666304, -1.3036803 , -0.27401102],
            [ 0.9138413 , -1.42192   , -0.25162885,  0.25845474, -0.67666304,  3.84462   ,  0.01445045, -2.6811552 ],
            [ 0.80170834, -0.08692162, -1.7232978 , -1.1731135 , -1.3036803 ,  0.01445043,  2.2252698 ,  1.2455846 ],
            [ 0.20153992,  1.9408858 , -1.1731135 , -2.4828184 , -0.27401105, -2.6811552 ,  1.2455845 ,  3.223088  ]
        ]), rtol=1e-5, atol=1e-5)

        np.testing.assert_allclose(residuals.flatten(), jnp.array([
            -0.4947916, -0.9895833, -0.3645833, -0.7291666, 
            -0.42708337, -0.8541667, -0.55729157, -1.1145831
        ]), rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()