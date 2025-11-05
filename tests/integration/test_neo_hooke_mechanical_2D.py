import pytest
import unittest
import optax
from flax import nnx     
import jax 
import os
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.explicit_parametric_operator_learning import ExplicitParametricOperatorLearning
from fol.tools.usefull_functions import *

class TestMechanicalNL2D(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    def setUp(self):
        # problem setup
        test_name = 'test_mechanical_NL_2D'
        self.test_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_name)
        create_clean_directory(self.test_directory)

        self.fe_mesh = create_2D_square_mesh(L=1,N=11)
        bc_dict = {"Ux":{"left":0.0,"right":0.1},
                   "Uy":{"left":0.0,"right":0.1}}
        
        material_dict = {"young_modulus":1,"poisson_ratio":0.3}
        self.mechanical_loss = NeoHookeMechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "num_gp":2,
                                                                                "material_dict":material_dict},
                                                                                fe_mesh=self.fe_mesh)
        fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"},
                      "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                    "maxiter":5,"load_incr":5}}
        self.fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",self.mechanical_loss,fe_setting)
        fourier_control_settings = {"x_freqs":np.array([1,2,3]),"y_freqs":np.array([1,2,3]),"z_freqs":np.array([0]),"beta":2}
        self.fourier_control = FourierControl("fourier_control",fourier_control_settings,self.fe_mesh)

        self.fe_mesh.Initialize()
        self.mechanical_loss.Initialize()
        self.fourier_control.Initialize()

        # design NN for learning
        class MLP(nnx.Module):
            def __init__(self, in_features: int, dmid: int, out_features: int, *, rngs: nnx.Rngs):
                self.dense1 = nnx.Linear(in_features, dmid, rngs=rngs,kernel_init=nnx.initializers.zeros,bias_init=nnx.initializers.zeros)
                self.dense2 = nnx.Linear(dmid, out_features, rngs=rngs,kernel_init=nnx.initializers.zeros,bias_init=nnx.initializers.zeros)
                self.in_features = in_features
                self.out_features = out_features

            def __call__(self, x: jax.Array) -> jax.Array:
                x = self.dense1(x)
                x = jax.nn.swish(x)
                x = self.dense2(x)
                return x

        fol_net = MLP(self.fourier_control.GetNumberOfVariables(),1, 
                        self.mechanical_loss.GetNumberOfUnknowns(), 
                        rngs=nnx.Rngs(0))

        # create fol optax-based optimizer
        chained_transform = optax.chain(optax.normalize_by_update_norm(), 
                                        optax.adam(1e-3))

        # create fol
        self.fol = ExplicitParametricOperatorLearning(name="dis_fol",control=self.fourier_control,
                                                        loss_function=self.mechanical_loss,
                                                        flax_neural_network=fol_net,
                                                        optax_optimizer=chained_transform)

        self.fol.Initialize()
        self.fe_solver.Initialize()
        self.coeffs_matrix,self.K_matrix = create_random_fourier_samples(self.fourier_control,0)

    def test_compute(self):

        self.fol.Train(train_set=(self.coeffs_matrix[-1].reshape(-1,1).T,),
                       convergence_settings={"num_epochs":200},
                       working_directory=self.test_directory)

        UV_FOL = np.array(self.fol.Predict(self.coeffs_matrix[-1,:].reshape(-1,1).T)).reshape(-1)
        UV_FEM = np.array(self.fe_solver.Solve(self.K_matrix[-1,:],np.zeros(UV_FOL.shape)))
        l2_error = 100 * np.linalg.norm(UV_FOL-UV_FEM,ord=2)/ np.linalg.norm(UV_FEM,ord=2)
        self.assertLessEqual(l2_error, 10)

        if self.debug_mode=="false":
            shutil.rmtree(self.test_directory)
        else:
            plot_mesh_vec_data(1,[self.K_matrix[-1,:],UV_FOL[0::2],UV_FOL[1::2]],["K","U","V"],file_name=os.path.join(self.test_directory,"FOL-KUV-dist.png"))
            plot_mesh_vec_data(1,[self.K_matrix[-1,:],UV_FEM[0::2],UV_FEM[1::2]],["K","U","V"],file_name=os.path.join(self.test_directory,"FEM-KUV-dist.png"))
            self.fe_mesh['K'] = np.array(self.K_matrix[-1,:])
            self.fe_mesh['UV_FOL'] = np.array(UV_FOL).reshape((self.fe_mesh.GetNumberOfNodes(),2))
            self.fe_mesh['UV_FEM'] = np.array(UV_FEM).reshape((self.fe_mesh.GetNumberOfNodes(),2))
            self.fe_mesh.Finalize(export_dir=self.test_directory)

if __name__ == '__main__':
    unittest.main()