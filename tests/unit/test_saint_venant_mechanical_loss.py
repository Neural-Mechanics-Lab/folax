import pytest
import unittest 
import os,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import numpy as np
from fol.loss_functions.mechanical_saint_venant import SaintVenantMechanicalLoss2DQuad
from fol.loss_functions.mechanical_saint_venant_AD import SaintVenantMechanicalLoss2DQuad as SaintVenantMechanicalLoss2DQuadAD
from fol.loss_functions.mechanical_saint_venant import SaintVenantMechanicalLoss3DTetra
from fol.loss_functions.mechanical_saint_venant_AD import SaintVenantMechanicalLoss3DTetra as SaintVenantMechanicalLoss3DTetraAD
from fol.loss_functions.mechanical_saint_venant import SaintVenantMechanicalLoss3DHexa
from fol.loss_functions.mechanical_saint_venant_AD import SaintVenantMechanicalLoss3DHexa as SaintVenantMechanicalLoss3DHexaAD
from fol.tools.usefull_functions import *

class TestMechanical3D(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def test_tetra(self):

        tet_points_coordinates = jnp.array([[0.1, 0.1, 0.1],
                                            [0.28739360416666665, 0.27808503701741405, 0.05672979583333333],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 1.0, 0.1]])

        fe_mesh = Mesh("",".")
        fe_mesh.node_ids = jnp.arange(len(tet_points_coordinates))
        fe_mesh.nodes_coordinates = tet_points_coordinates
        fe_mesh.elements_nodes = {"tetra":fe_mesh.node_ids.reshape(1,-1)}

        mechanical_loss_3d = SaintVenantMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":{"Ux":{},"Uy":{},"Uz":{}},
                                                                                       "material_dict":{"young_modulus":1,"poisson_ratio":0.3},
                                                                                       "body_foce":jnp.array([[1],[2],[3]])},
                                                                                        fe_mesh=fe_mesh)
        mechanical_loss_3d.Initialize()
        
        en, residuals, stiffness = mechanical_loss_3d.ComputeElement(tet_points_coordinates,
                                                                        jnp.ones((4)),
                                                                        jnp.ones((12,1)))
        
        mechanical_loss_AD_3d = SaintVenantMechanicalLoss3DTetraAD("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":{"Ux":{},"Uy":{},"Uz":{}},
                                                                                       "material_dict":{"young_modulus":1,"poisson_ratio":0.3},
                                                                                       "body_foce":jnp.array([[1],[2],[3]])},
                                                                                        fe_mesh=fe_mesh)
        mechanical_loss_AD_3d.Initialize()
        
        en_ad, residuals_ad, stiffness_ad = mechanical_loss_AD_3d.ComputeElement(tet_points_coordinates,
                                                                        jnp.ones((4)),
                                                                        jnp.ones((12,1)))

        
        np.testing.assert_allclose(stiffness,stiffness_ad,rtol=1e-5, atol=1e-6)

        np.testing.assert_allclose(residuals,residuals_ad,rtol=1e-5, atol=1e-6)

    def test_hexa(self):
        hex_points_coordinates = jnp.array([[0.24900,  0.34200,  0.19200],
                                            [0.32000,  0.18600,  0.64300],
                                            [0.16500,  0.74500,  0.70200],
                                            [0.27300,  0.75000,  0.23000],
                                            [0.00000,  0.00000,  0.00000],
                                            [0.00000,  0.00000,  1.00000],
                                            [0.00000,  1.00000,  1.00000],
                                            [0.00000,  1.00000,  0.00000]])

        fe_mesh = Mesh("",".")
        fe_mesh.node_ids = jnp.arange(len(hex_points_coordinates))
        fe_mesh.nodes_coordinates = hex_points_coordinates
        fe_mesh.elements_nodes = {"hexahedron":fe_mesh.node_ids.reshape(1,-1)}

        mechanical_loss_3d = SaintVenantMechanicalLoss3DHexa("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":{"Ux":{},"Uy":{},"Uz":{}},
                                                  "material_dict":{"young_modulus":1,"poisson_ratio":0.3},
                                                  "body_foce":jnp.array([[1],[2],[3]])},
                                                   fe_mesh=fe_mesh)
        mechanical_loss_3d.Initialize()

        en, residuals, stiffness = mechanical_loss_3d.ComputeElement(hex_points_coordinates,
                                                                        jnp.ones((8)),
                                                                        jnp.ones((24,1)))
        
        mechanical_loss_AD_3d = SaintVenantMechanicalLoss3DHexaAD("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":{"Ux":{},"Uy":{},"Uz":{}},
                                                  "material_dict":{"young_modulus":1,"poisson_ratio":0.3},
                                                  "body_foce":jnp.array([[1],[2],[3]])},
                                                   fe_mesh=fe_mesh)
        mechanical_loss_AD_3d.Initialize()

        en_ad, residuals_ad, stiffness_ad = mechanical_loss_AD_3d.ComputeElement(hex_points_coordinates,
                                                                        jnp.ones((8)),
                                                                        jnp.ones((24,1)))

        np.testing.assert_allclose(stiffness,stiffness_ad,
                                    rtol=1e-5, atol=1e-6)
        
        np.testing.assert_allclose(residuals,residuals_ad,
                                    rtol=1e-5, atol=1e-6)
        
    def test_quad(self):
                
        quad_points_coordinates = jnp.array([[3.00,0.00,0.00],
                                            [2.00,0.75,0.00],
                                            [0.75,1.00,0.00],
                                            [0.00,0.00,0.00]])
        
        fe_mesh = Mesh("",".")
        fe_mesh.node_ids = jnp.arange(len(quad_points_coordinates))
        fe_mesh.nodes_coordinates = quad_points_coordinates
        fe_mesh.elements_nodes = {"quad":fe_mesh.node_ids.reshape(1,-1)}

        mechanical_loss_3d = SaintVenantMechanicalLoss2DQuad("mechanical_loss_2d",
                                                  loss_settings={"dirichlet_bc_dict":{"Ux":{},"Uy":{},"Uz":{}},
                                                                 "material_dict":{"young_modulus":1, "poisson_ratio":0.3},
                                                                 "body_foce":jnp.array([[1],[2]])},
                                                                 fe_mesh=fe_mesh)
        mechanical_loss_3d.Initialize()

        en, residuals, stiffness = mechanical_loss_3d.ComputeElement(quad_points_coordinates,
                                                                     jnp.ones((4)),
                                                                     jnp.ones((8,1)))
        
        mechanical_loss_AD_3d = SaintVenantMechanicalLoss2DQuad("mechanical_loss_2d",
                                                  loss_settings={"dirichlet_bc_dict":{"Ux":{},"Uy":{},"Uz":{}},
                                                                 "material_dict":{"young_modulus":1, "poisson_ratio":0.3},
                                                                 "body_foce":jnp.array([[1],[2]])},
                                                                 fe_mesh=fe_mesh)
        mechanical_loss_AD_3d.Initialize()

        en_ad, residuals_ad, stiffness_ad = mechanical_loss_AD_3d.ComputeElement(quad_points_coordinates,
                                                                     jnp.ones((4)),
                                                                     jnp.ones((8,1)))

        np.testing.assert_allclose(stiffness,stiffness_ad,
                                   rtol=1e-5, atol=1e-6)

        np.testing.assert_allclose(residuals,residuals_ad,
                                   rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()