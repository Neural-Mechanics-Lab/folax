import pytest
import unittest
import optax
from flax import nnx     
import jax 
import os
import numpy as np
from fol.loss_functions.mechanical import MechanicalLoss3DTetra
from fol.loss_functions.mechanical import MechanicalLoss3DHexa
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.controls.voronoi_control3D import VoronoiControl3D
from fol.deep_neural_networks.explicit_parametric_operator_learning import ExplicitParametricOperatorLearning
from fol.tools.usefull_functions import *

class TestMechanical3D(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def _request_debug_mode(self,request):
        self.debug_mode = request.config.getoption('--debug-mode')

    @classmethod
    def setUpClass(cls):
        # problem setup for tetrahedron
        test_tetra_name = 'test_mechanical_3D_poly_lin_tetra'
        cls.test_tetra_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_tetra_name)
        create_clean_directory(cls.test_tetra_directory)

        # problem setup for hexahedron
        test_hexa_name = 'test_mechanical_3D_poly_lin_hexa'
        cls.test_hexa_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), test_hexa_name)
        create_clean_directory(cls.test_hexa_directory)

        cls.dirichlet_bc_dict = {"Ux":{"left":0.0},
                "Uy":{"left":0.0,"right":-0.05},
                "Uz":{"left":0.0,"right":-0.05}}
        
        cls.fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"},
                      "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                    "maxiter":5,"load_incr":5}}
        

    def test_compute_tetrahedron(self):
        fe_mesh = Mesh("box_io","box_3D_coarse.med",os.path.join(os.path.dirname(os.path.abspath(__file__)),"../meshes"))
        mechanical_loss_3d = MechanicalLoss3DTetra("mechanical_tetra_loss_3d",loss_settings={"dirichlet_bc_dict":self.dirichlet_bc_dict,
                                                                                            "material_dict":{"young_modulus":1,"poisson_ratio":0.3}},
                                                                                            fe_mesh=fe_mesh)
        linear_fe_solver = FiniteElementLinearResidualBasedSolver("lin_fe_solver",mechanical_loss_3d,self.fe_setting)

        voronoi_control_settings = {"number_of_seeds":16,"E_values":[0.1,1]}
        voronoi_control = VoronoiControl3D("voronoi_control",voronoi_control_settings,fe_mesh)

        fe_mesh.Initialize()
        mechanical_loss_3d.Initialize()
        voronoi_control.Initialize()

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

        fol_net = MLP(voronoi_control.GetNumberOfVariables(),1, 
                        mechanical_loss_3d.GetNumberOfUnknowns(), 
                        rngs=nnx.Rngs(0))

        # create fol optax-based optimizer
        chained_transform = optax.chain(optax.normalize_by_update_norm(), 
                                        optax.adam(1e-4))

        # create fol
        fol = ExplicitParametricOperatorLearning(name="dis_fol",control=voronoi_control,
                                                        loss_function=mechanical_loss_3d,
                                                        flax_neural_network=fol_net,
                                                        optax_optimizer=chained_transform)

        fol.Initialize()
        linear_fe_solver.Initialize()        
        coeffs_matrix,K_matrix = create_random_voronoi_samples(voronoi_control,1,dim=3)
        fol.Train(train_set=(coeffs_matrix[-1].reshape(-1,1).T,),
                       convergence_settings={"num_epochs":1000,"relative_error":1e-12,"absolute_error":1e-12},
                       working_directory=self.test_tetra_directory)
        U_FOL = np.array(fol.Predict(coeffs_matrix.reshape(-1,1).T)).reshape(-1)
        U_FEM = np.array(linear_fe_solver.Solve(K_matrix.flatten(),np.zeros(U_FOL.shape)))
        l2_error = 100 * np.linalg.norm(U_FOL-U_FEM,ord=2)/ np.linalg.norm(U_FEM,ord=2)
        self.assertLessEqual(l2_error, 10)

        if self.debug_mode=="false":
            shutil.rmtree(self.test_tetra_directory)
        else:
            fe_mesh['K'] = np.array(K_matrix)
            fe_mesh['U_FOL'] = np.array(U_FOL).reshape((fe_mesh.GetNumberOfNodes(),3))
            fe_mesh['U_FEM'] = np.array(U_FEM).reshape((fe_mesh.GetNumberOfNodes(),3))
            fe_mesh.Finalize(export_dir=self.test_tetra_directory)

    def test_compute_hexahedron(self):
        fe_mesh = create_3D_box_mesh(Nx=11,Ny=11,Nz=11,Lx=1.,Ly=1.,Lz=1.,case_dir=self.test_hexa_directory)
        mechanical_loss_3d = MechanicalLoss3DHexa("mechanical_hexa_loss_3d",loss_settings={"dirichlet_bc_dict":self.dirichlet_bc_dict,
                                                                                            "material_dict":{"young_modulus":1,"poisson_ratio":0.3}},
                                                                                            fe_mesh=fe_mesh)
        linear_fe_solver = FiniteElementLinearResidualBasedSolver("lin_fe_solver",mechanical_loss_3d,self.fe_setting)

        voronoi_control_settings = {"number_of_seeds":16,"E_values":[0.1,1]}
        voronoi_control = VoronoiControl3D("voronoi_control",voronoi_control_settings,fe_mesh)

        fe_mesh.Initialize()
        mechanical_loss_3d.Initialize()
        voronoi_control.Initialize()

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

        fol_net = MLP(voronoi_control.GetNumberOfVariables(),1, 
                        mechanical_loss_3d.GetNumberOfUnknowns(), 
                        rngs=nnx.Rngs(0))

        # create fol optax-based optimizer
        chained_transform = optax.chain(optax.normalize_by_update_norm(), 
                                        optax.adam(1e-4))

        # create fol
        fol = ExplicitParametricOperatorLearning(name="dis_fol",control=voronoi_control,
                                                        loss_function=mechanical_loss_3d,
                                                        flax_neural_network=fol_net,
                                                        optax_optimizer=chained_transform)

        fol.Initialize()
        linear_fe_solver.Initialize()        
        coeffs_matrix,K_matrix = create_random_voronoi_samples(voronoi_control,1,dim=3)
        fol.Train(train_set=(coeffs_matrix[-1].reshape(-1,1).T,),
                       convergence_settings={"num_epochs":1000,"relative_error":1e-12,"absolute_error":1e-12},
                       working_directory=self.test_hexa_directory)
        U_FOL = np.array(fol.Predict(coeffs_matrix.reshape(-1,1).T)).reshape(-1)
        U_FEM = np.array(linear_fe_solver.Solve(K_matrix.flatten(),np.zeros(U_FOL.shape)))
        l2_error = 100 * np.linalg.norm(U_FOL-U_FEM,ord=2)/ np.linalg.norm(U_FEM,ord=2)
        self.assertLessEqual(l2_error, 10)

        if self.debug_mode=="false":
            shutil.rmtree(self.test_hexa_directory)
        else:
            fe_mesh['K'] = np.array(K_matrix)
            fe_mesh['U_FOL'] = np.array(U_FOL).reshape((fe_mesh.GetNumberOfNodes(),3))
            fe_mesh['U_FEM'] = np.array(U_FEM).reshape((fe_mesh.GetNumberOfNodes(),3))
            fe_mesh.Finalize(export_dir=self.test_hexa_directory)

if __name__ == '__main__':
    unittest.main()