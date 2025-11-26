import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import optax
import numpy as np
from fol.loss_functions.regression_loss import RegressionLoss
from fol.controls.identity_control import IdentityControl
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DTetra
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from data_driven_meta_alpha_meta_implicit_parametric_operator_learning import DataDrivenMetaAlphaMetaImplicitParametricOperatorLearning
from data_driven_meta_implicit_parametric_operator_learning import DataDrivenMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
# from fol.data_input_output.zarr_io import ZarrIO
from fol.controls.dirichlet_control import DirichletControl
import pickle

# directory & save handling
working_directory_name = 'nn_output_mechanical_3d_data_driven'
case_dir = os.path.join('.', working_directory_name)
# create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

#call the function to create the mesh
fe_mesh = Mesh("fol_io","cylindrical_fine.med",os.path.join(os.path.dirname(__file__),'..','..','meshes'))
fe_mesh.Initialize()

# creation of fe model and loss function
bc_dict = {"Ux":{"left":0.0,"right":0.25},
            "Uy":{"left":0.0,"right":0.25},
            "Uz":{"left":0.0,"right":0.0}}
material_dict = {"young_modulus":1,"poisson_ratio":0.3}
loss_settings = {"dirichlet_bc_dict":bc_dict,"parametric_boundary_learning":True,"material_dict":material_dict}
mechanical_loss_3d = NeoHookeMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings=loss_settings,
                                                                                fe_mesh=fe_mesh)
mechanical_loss_3d.Initialize()


# dirichlet control
dirichlet_control_settings = {"learning_boundary":{"Ux":{'right'},"Uy":{"right"}, "Uz":{"right"}}}
dirichlet_control = DirichletControl(control_name='dirichlet_control',control_settings=dirichlet_control_settings, 
                                        fe_mesh= fe_mesh,fe_loss=mechanical_loss_3d)
dirichlet_control.Initialize()


coeffs_matrix_test = np.array([[-0.05,0.2,0.15],
                                [0.25,0.15,0.2],
                                [0.2,-0.05,0.1]])

### extend K_matrix to incorporate all dofs
K_mat = []
K_matrix_dirichlet = dirichlet_control.ComputeBatchControlledVariables(coeffs_matrix_test)

for i in range(coeffs_matrix_test.shape[0]):
    K_matrix_total = np.zeros(3*fe_mesh.GetNumberOfNodes())
    K_matrix_total[mechanical_loss_3d.dirichlet_indices]= K_matrix_dirichlet[i,:]
    K_mat.append(K_matrix_total)
K_matrix = np.array(K_mat)
print(K_matrix.shape)

# create identity control
identity_control = IdentityControl("ident_control",control_settings={},num_vars=3*fe_mesh.GetNumberOfNodes())

# create regression loss
reg_loss = RegressionLoss("reg_loss",loss_settings={"nodal_unknows":["Ux","Uy","Uz"]},fe_mesh=fe_mesh)

# initialize all 
reg_loss.Initialize()
identity_control.Initialize()

# design siren NN for learning
characteristic_length = 64
synthesizer_nn = MLP(name="regressor_synthesizer",
                    input_size=3,
                    output_size=3,
                    hidden_layers=[characteristic_length] * 4,
                    activation_settings={"type":"sin",
                                         "prediction_gain":30,
                                         "initialization_gain":1.0})

latent_size = 64
modulator_nn = MLP(name="modulator_nn",
                   input_size=latent_size,
                   use_bias=False) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
num_epochs = 20000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(1e-5))
latent_step_optimizer = optax.chain(optax.normalize_by_update_norm(),optax.adam(1e-5))

# create data driven class of Meta Implicit Operator Learning
fol = DataDrivenMetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=identity_control,
                                            loss_function=reg_loss,
                                            flax_neural_network=hyper_network,
                                            main_loop_optax_optimizer=main_loop_transform,
                                            latent_step_size=1e-2,
                                            num_latent_iterations=3)

fol.Initialize()

# load the best model
fol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

for eval_id in range(coeffs_matrix_test.shape[0]):
    U_iFOL = np.array(fol.Predict(K_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
    
    updated_bc = bc_dict.copy()
    updated_bc.update({"Ux":{"left":0.,"right":coeffs_matrix_test[eval_id,0]},
                        "Uy":{"left":0.,"right":coeffs_matrix_test[eval_id,1]},
                        "Uz":{"left":0.,"right":coeffs_matrix_test[eval_id,2]}})

    updated_loss_setting = loss_settings.copy()
    updated_loss_setting.update({"dirichlet_bc_dict":updated_bc})
    mechanical_loss_3d_updated = NeoHookeMechanicalLoss3DTetra("mechanical_loss_3d",loss_settings=updated_loss_setting,
                                                                                fe_mesh=fe_mesh)
    mechanical_loss_3d_updated.Initialize()

    fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                                "maxiter":10,"load_incr":1}}
    nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_3d_updated,fe_setting)
    nonlin_fe_solver.Initialize()
    try:
        U_FEM = np.array(nonlin_fe_solver.Solve(np.ones(fe_mesh.GetNumberOfNodes()),U_iFOL.reshape(3*fe_mesh.GetNumberOfNodes())))
    except Exception as e:
        print(f"Error value: {e}")        
        U_FEM = np.zeros((3,fe_mesh.GetNumberOfNodes()))
    abs_err = abs(U_iFOL.reshape(-1)-U_FEM.reshape(-1))
    fe_mesh[f'U_FEM_{eval_id}'] = U_FEM.reshape((fe_mesh.GetNumberOfNodes(), 3))
    fe_mesh[f'U_iFOL_{eval_id}'] = U_iFOL.reshape((fe_mesh.GetNumberOfNodes(), 3))
    fe_mesh[f'abs_err_{eval_id}'] = abs_err.reshape((fe_mesh.GetNumberOfNodes(), 3))


fe_mesh.Finalize(export_dir=case_dir,export_format='vtu')