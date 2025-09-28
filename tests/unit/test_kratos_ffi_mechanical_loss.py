import os
import pytest
import unittest 
import importlib
import numpy as np
from fol.tools.usefull_functions import *
import jax

def package_and_gpu_available():
    # Check if the fol_ffi_functions exists
    package_exists = importlib.util.find_spec("fol_ffi_functions") is not None
    # Check if at least one GPU is available
    gpu_exists = any(d.platform == "gpu" for d in jax.devices())
    return package_exists and gpu_exists

@unittest.skipUnless(package_and_gpu_available(), "Requires fol_ffi_functions lib and a GPU")
class TestKratosFFIMechanical3D(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # Skip test if no GPU is available
        if not any(d.platform == "gpu" for d in jax.devices()):
            pytest.skip("Skipping TestKratosFFIMechanical3D: no GPU available for testing")

    def test_tetra(self):

        tet_points_coordinates = jnp.array([[0.1, 0.1, 0.1],
                                            [0.28739360416666665, 0.27808503701741405, 0.05672979583333333],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 1.0, 0.1]])

        fe_mesh = Mesh("",".")
        fe_mesh.node_ids = jnp.arange(len(tet_points_coordinates))
        fe_mesh.nodes_coordinates = tet_points_coordinates
        fe_mesh.elements_nodes = {"tetra":fe_mesh.node_ids.reshape(1,-1)}

        from fol.loss_functions.kratos_small_displacement import KratosSmallDisplacement3DTetra
        mechanical_loss_3d = KratosSmallDisplacement3DTetra("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":{"Ux":{},"Uy":{},"Uz":{}},
                                                                                       "material_dict":{"young_modulus":1,"poisson_ratio":0.3},
                                                                                       "body_foce":jnp.array([[0],[0],[0]])},
                                                                                        fe_mesh=fe_mesh)
        mechanical_loss_3d.Initialize()

        # generate batch of random solution field
        key = jax.random.PRNGKey(42)
        batch_nodal_solution = jax.random.uniform(key, shape=(5, 12))

        # test energy
        batch_energies = jax.vmap(mechanical_loss_3d.ComputeTotalEnergy)(batch_nodal_solution,batch_nodal_solution)
        np.testing.assert_allclose(batch_energies.flatten(),jnp.array([0.11373918, 0.09218077, 0.02612576, 0.03834702, 0.09644907]),rtol=1e-5, atol=1e-5)

        # test residuals amd jac
        jac,res = mechanical_loss_3d.ComputeJacobianMatrixAndResidualVector(batch_nodal_solution[0].flatten(),batch_nodal_solution[0].flatten())
        np.testing.assert_allclose(res.flatten(),jnp.array([-0.03185109794139862, 
                                                            -0.00037236514617688954, 
                                                            -0.02698179893195629, 
                                                            0.04158858209848404, 
                                                            -0.004593688528984785, 
                                                            0.040204968303442, 
                                                            -0.10656017065048218, 
                                                            0.04939720034599304, 
                                                            -0.10868339240550995, 
                                                            0.09682268649339676, 
                                                            -0.04443114623427391, 
                                                            0.09546022862195969]),rtol=1e-5, atol=1e-5)

        np.testing.assert_allclose(jac.todense().flatten(),jnp.array([0.0655517578125, 0.0178375244140625, 0.0, -0.07916259765625, -0.01261138916015625, 
                                                                      0.0, 0.03424072265625, 0.005458831787109375, 0.06939697265625, -0.0206298828125, 
                                                                      -0.01067352294921875, -0.06939697265625, 0.0178375244140625, 0.0278472900390625, 
                                                                      0.0, -0.0158233642578125, -0.025787353515625, 0.0, 0.00684356689453125, 0.01116180419921875, 
                                                                      0.0276336669921875, -0.00885772705078125, -0.01322174072265625, -0.0276336669921875, 
                                                                      0.0, 0.0, 0.020751953125, 0.0, 0.0, -0.0233306884765625, 0.0462646484375, 0.0184173583984375, 
                                                                      0.01009368896484375, -0.0462646484375, -0.0184173583984375, -0.0075225830078125, -0.07916259765625, 
                                                                      -0.0158233642578125, 0.0, 0.0977783203125, 0.0077362060546875, 0.0, -0.042327880859375, -0.0033473968505859375, 
                                                                      -0.0865478515625, 0.023681640625, 0.0114288330078125, 0.0865478515625, -0.01261138916015625, -0.025787353515625, 
                                                                      0.0, 0.0077362060546875, 0.029052734375, 0.0, -0.0033473968505859375, -0.0125732421875, -0.009613037109375, 0.0082244873046875, 
                                                                      0.00931549072265625, 0.009613037109375, 0.0, 0.0, -0.0233306884765625, 0.0, 0.0, 0.0281829833984375, -0.05767822265625, 
                                                                      -0.00640869140625, -0.01219940185546875, 0.05767822265625, 0.00640869140625, 0.00733184814453125, 0.03424072265625, 0.00684356689453125, 
                                                                      0.0462646484375, -0.042327880859375, -0.0033473968505859375, -0.05767822265625, 0.1378173828125, 0.0014486312866210938, 0.062408447265625, 
                                                                      -0.1297607421875, -0.004947662353515625, -0.050994873046875, 0.005458831787109375, 0.01116180419921875, 0.0184173583984375, -0.0033473968505859375, 
                                                                      -0.0125732421875, -0.00640869140625, 0.0014486312866210938, 0.12493896484375, 0.00693511962890625, -0.003559112548828125, -0.12353515625, -0.018951416015625, 
                                                                      0.06939697265625, 0.0276336669921875, 0.01009368896484375, -0.0865478515625, -0.009613037109375, -0.01219940185546875, 0.062408447265625, 0.00693511962890625,
                                                                      0.423583984375, -0.0452880859375, -0.0249481201171875, -0.421630859375, -0.0206298828125, -0.00885772705078125, -0.0462646484375, 0.023681640625, 0.0082244873046875,
                                                                      0.05767822265625, -0.1297607421875, -0.003559112548828125, -0.0452880859375, 0.126708984375, 0.00418853759765625, 0.03387451171875, -0.01067352294921875, -0.01322174072265625, 
                                                                      -0.0184173583984375, 0.0114288330078125, 0.00931549072265625, 0.00640869140625, -0.004947662353515625, -0.12353515625, -0.0249481201171875, 0.00418853759765625, 0.12744140625, 
                                                                      0.036956787109375, -0.06939697265625, -0.0276336669921875, -0.0075225830078125, 0.0865478515625, 0.009613037109375, 0.00733184814453125, -0.050994873046875, -0.018951416015625, 
                                                                      -0.421630859375, 0.03387451171875, 0.036956787109375, 0.421630859375]),rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
