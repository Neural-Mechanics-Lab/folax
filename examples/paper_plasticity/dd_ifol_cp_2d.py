import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import optax
import numpy as np
from fol.loss_functions.regression_loss import RegressionLoss
from fol.controls.identity_control import IdentityControl
from data_driven_meta_alpha_meta_implicit_parametric_operator_learning import DataDrivenMetaAlphaMetaImplicitParametricOperatorLearning
from data_driven_meta_implicit_parametric_operator_learning import DataDrivenMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DTetra
import pickle

# directory & save handling
working_directory_name = 'ifol_output_cp_2d_mata_implicit'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":128,
                    "Ux_left":0.0,"Ux_right":0.,
                    "Uy_left":0.0,"Uy_right":0.}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()


with open(os.path.join(os.path.dirname(__file__),'ifol_output_gt_cp/damask_concatenate_data.pkl'), 'rb') as f:
    gt_dict = pickle.load(f)

increments = [75, 150, 225, 300, 375, 450, 525, 600, 675, 750]
sim_num = 100
channel_num = 5
N = model_settings["N"]
row = int(sim_num*len(increments))
col = int(N*N*channel_num)    # nx=128, ny=128, number of variable/channel = 5
K_matrix_current_step = np.zeros((row,col))
K_matrix_next_step = np.zeros((row,col))

for i, incr in enumerate(increments[:-1]):
    K_matrix_current_step[i:sim_num+i,:] = gt_dict[f'increment_{incr}']
    K_matrix_next_step[i:sim_num+i,:] = gt_dict[f'increment_{incr+75}']

# normalize data
max_orientation = max(np.max(K_matrix_current_step[:,:-1]),np.max(K_matrix_next_step[:,:-1]))
max_stress = max(np.max(K_matrix_current_step[:,-1]),np.max(K_matrix_next_step[:,-1]))
K_matrix_current_step[:,:-1], K_matrix_next_step[:,:-1] = K_matrix_current_step[:,:-1] / max_orientation, K_matrix_next_step[:,:-1] / max_orientation
K_matrix_current_step[:,-1], K_matrix_next_step[:,-1] = K_matrix_current_step[:,-1] / max_stress, K_matrix_next_step[:,-1] / max_stress


# create identity control
identity_control = IdentityControl("ident_control",num_vars=N*N*channel_num)

# create regression loss
reg_loss = RegressionLoss("reg_loss",loss_settings={"nodal_unknows":["Ori_0","Ori_1","Ori_2","Ori_3","von_mises"]},fe_mesh=fe_mesh)

# initialize all 
reg_loss.Initialize()
identity_control.Initialize()

# design siren NN for learning
characteristic_length = 64
synthesizer_nn = MLP(name="regressor_synthesizer",
                    input_size=3,
                    output_size=5,
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
num_epochs = 120
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(1e-5))
latent_step_optimizer = optax.chain(optax.normalize_by_update_norm(),optax.adam(1e-5))

# create fol
meta_learning = False
if meta_learning:
    fol = DataDrivenMetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=identity_control,
                                                loss_function=reg_loss,
                                                flax_neural_network=hyper_network,
                                                main_loop_optax_optimizer=main_loop_transform,
                                                latent_step_size=1e-2,
                                                latent_step_optax_optimizer=latent_step_optimizer,
                                                num_latent_iterations=3)

else:
    fol = DataDrivenMetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=identity_control,
                                                loss_function=reg_loss,
                                                flax_neural_network=hyper_network,
                                                main_loop_optax_optimizer=main_loop_transform,
                                                latent_step_size=1e-2,
                                                num_latent_iterations=3)

fol.Initialize()

train_start_id = 0
train_end_id = 9 * sim_num
test_start_id = 9 * sim_num
test_end_id = 9 * sim_num + 10

fol.Train(train_set=(K_matrix_current_step[train_start_id:train_end_id,:],K_matrix_next_step[train_start_id:train_end_id,:]),
          test_set=(K_matrix_current_step[test_start_id:test_end_id,:],K_matrix_next_step[test_start_id:test_end_id,:]),
          test_frequency=10,batch_size=5,
          convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
          train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
          working_directory=case_dir)

# load the best model
fol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

for eval_id in list(np.arange(test_start_id,test_end_id)):
    iFOL_next_step = np.array(fol.Predict(K_matrix_current_step[eval_id,:].reshape(-1,1).T)).reshape(-1)
    iFOL_von_mises_next_step = iFOL_next_step[::5]
    von_mises_gt_next = K_matrix_next_step[eval_id,::5]
    abs_err = abs(iFOL_von_mises_next_step.reshape(-1)-von_mises_gt_next.reshape(-1))
    fe_mesh[f'Von_Mises_FEM_{eval_id}'] = iFOL_von_mises_next_step.reshape((fe_mesh.GetNumberOfNodes(), 1))
    fe_mesh[f'Von_Mises_iFOL_{eval_id}'] = von_mises_gt_next.reshape((fe_mesh.GetNumberOfNodes(), 1))
    fe_mesh[f'abs_err_{eval_id}'] = abs_err.reshape((fe_mesh.GetNumberOfNodes(), 1))


fe_mesh.Finalize(export_dir=case_dir,export_format='vtu')