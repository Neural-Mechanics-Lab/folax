import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import optax
import numpy as np
from fol.loss_functions.regression_loss import RegressionLoss
from fol.controls.identity_control import IdentityControl
from data_driven_meta_alpha_meta_implicit_parametric_operator_learning import DataDrivenMetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.data_input_output.zarr_io import ZarrIO
from dirichlet_control import DirichletControl3D
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DTetra

# directory & save handling
working_directory_name = 'meta_alpha_meta_implicit_operator_learning'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

#call the function to create the mesh
fe_mesh = Mesh("fol_io","cylindrical_fine_all_sides.med",os.path.join(os.path.abspath(__file__),'../../../meshes'))
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
dirichlet_control = DirichletControl3D(control_name='dirichlet_control',control_settings=dirichlet_control_settings, 
                                        fe_mesh= fe_mesh,fe_loss=mechanical_loss_3d)
dirichlet_control.Initialize()


# create some random coefficients & K for training
n_samples = 200
np.random.seed(42)
ux_comp = np.random.uniform(low=-0.1, high=0.4, size=n_samples).reshape(-1,1)
uy_comp = np.random.uniform(low=-0.1, high=0.1, size=n_samples).reshape(-1,1)
uz_comp = np.random.uniform(low=-0.1, high=0.1, size=n_samples).reshape(-1,1)
coeffs_matrix = np.concatenate((ux_comp,uy_comp),axis=1)
K_matrix = dirichlet_control.ComputeBatchControlledVariables(coeffs_matrix)

### First approach
# create identity control
identity_control = IdentityControl("ident_control",num_vars=len(mechanical_loss_3d.dirichlet_indices))

### Second approach
total_K_matrix = np.zeros(fe_mesh.GetNumberOfNodes())
total_K_matrix[mechanical_loss_3d.dirichlet_indices] = K_matrix
# create identity control
identity_control = IdentityControl("ident_control",num_vars=fe_mesh.GetNumberOfNodes())

# create regression loss
reg_loss = RegressionLoss("reg_loss",loss_settings={"nodal_unknows":["K"]},fe_mesh=fe_mesh)

# initialize all 
reg_loss.Initialize()
identity_control.Initialize()

# design siren NN for learning
characteristic_length = 64
synthesizer_nn = MLP(name="regressor_synthesizer",
                    input_size=3,
                    output_size=1,
                    hidden_layers=[characteristic_length] * 6,
                    activation_settings={"type":"sin",
                                         "prediction_gain":30,
                                         "initialization_gain":1.0})

latent_size = 10
modulator_nn = MLP(name="modulator_nn",
                   input_size=latent_size,
                   use_bias=False) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
num_epochs = 5000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(1e-5))
latent_step_optimizer = optax.chain(optax.normalize_by_update_norm(),optax.adam(1e-5))

# create fol
fol = DataDrivenMetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=identity_control,
                                                loss_function=reg_loss,
                                                flax_neural_network=hyper_network,
                                                main_loop_optax_optimizer=main_loop_transform,
                                                latent_step_size=1e-2,
                                                latent_step_optax_optimizer=latent_step_optimizer,
                                                num_latent_iterations=3)

fol.Initialize()

train_start_id = 0
train_end_id = 20
test_start_id = 3 * train_end_id
test_end_id = 4 * train_end_id

fol.Train(train_set=(data_sets["K"][train_start_id:train_end_id,:],data_sets["T_FEM"][train_start_id:train_end_id,:]),
          test_set=(data_sets["K"][test_start_id:test_end_id,:],data_sets["T_FEM"][test_start_id:test_end_id,:]),
          test_frequency=10,batch_size=5,
          convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
          train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
          working_directory=case_dir)

# load the best model
fol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

for eval_id in list(np.arange(train_start_id,test_end_id)):
    T_iFOL = np.array(fol.Predict(data_sets["K"][eval_id,:].reshape(-1,1).T)).reshape(-1)
    T_FEM = data_sets["T_FEM"][eval_id]
    K = data_sets["K"][eval_id]
    abs_err = abs(T_iFOL-T_FEM)
    plot_mesh_vec_data(1,[K,T_FEM,T_iFOL,abs_err],
                        ["conductivity","T_FEM","T_iFOL","abs_error"],
                        fig_title="",
                        file_name=os.path.join(case_dir,f"test_{eval_id}.png"))

