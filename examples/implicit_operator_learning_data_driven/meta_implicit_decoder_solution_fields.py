import sys
import os
import optax
import numpy as np
from fol.loss_functions.regression_loss import RegressionLoss
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.data_input_output.zarr_io import ZarrIO

# directory & save handling
working_directory_name = 'meta_implicit_decoder_solution_fields'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# import data sets if availbale otherwise run the generator and then import
if not os.path.exists("data_sets.zarr"):
    import subprocess
    process = subprocess.run(['python3', 'generate_data_sets_mechanical_2D.py'])
    # Check the return code to ensure the script ran successfully
    if process.returncode == 0:
        print("Script completed successfully")
    else:
        print("Script encountered an error")

data_sets = ZarrIO("zerr_io").Import("data_sets.zarr")
mesh_size = int(data_sets["T_FEM"].shape[1])

# problem setup
model_settings = {"L":1,"N":int(np.sqrt(mesh_size))}
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

# create identity control
identity_control = IdentityControl("ident_control",num_vars=mesh_size)

# create regression loss
reg_loss = RegressionLoss("reg_loss",loss_settings={"nodal_unknows":["T_FEM"]},fe_mesh=fe_mesh)

# initialize all 
reg_loss.Initialize()
identity_control.Initialize()

# design siren NN for learning
characteristic_length = model_settings["N"]
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

# create fol
fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=identity_control,
                                                loss_function=reg_loss,
                                                flax_neural_network=hyper_network,
                                                main_loop_optax_optimizer=main_loop_transform,
                                                latent_step_size=1e-2,
                                                num_latent_iterations=3)

fol.Initialize()

train_start_id = 0
train_end_id = 20
test_start_id = 3 * train_end_id
test_end_id = 4 * train_end_id

fol.Train(train_set=(data_sets["T_FEM"][train_start_id:train_end_id,:],),
          test_set=(data_sets["T_FEM"][test_start_id:test_end_id,:],),
          test_frequency=10,batch_size=1,
          convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
          train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
          working_directory=case_dir)

# load the best model
fol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

for eval_id in list(np.arange(train_start_id,test_end_id)):
    predicted = np.array(fol.Predict(data_sets["T_FEM"][eval_id,:].reshape(-1,1).T)).reshape(-1)
    ground_truth = data_sets["T_FEM"][eval_id]
    abs_err = abs(predicted-data_sets["T_FEM"][eval_id])
    plot_mesh_vec_data(1,[predicted,ground_truth,abs_err],
                        ["predicted","ground_truth","abs_error"],
                        fig_title="",
                        file_name=os.path.join(case_dir,f"test_{eval_id}.png"))

