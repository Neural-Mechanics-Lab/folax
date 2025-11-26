import sys
import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import optax
import numpy as np
from fol.loss_functions.regression_loss import RegressionLoss
from fol.controls.identity_control import IdentityControl
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.data_input_output.zarr_io import ZarrIO
import pickle,optax

# directory & save handling
working_directory_name = 'meta_implicit_decoder_conductivity_fields'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

mesh_size = 21

# problem setup
model_settings = {"L":1,"N":int((mesh_size))}
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

# create control
fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                            "beta":20,"min":1e-1,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

# create identity control
identity_control = IdentityControl("ident_control",num_vars=mesh_size)

# create regression loss
reg_loss = RegressionLoss("reg_loss",loss_settings={"nodal_unknows":["K"]},fe_mesh=fe_mesh)

# initialize all 
reg_loss.Initialize()
identity_control.Initialize()
fourier_control.Initialize()

# load some random coefficients & K for training
with open(f'fourier_3D_control_dict_10K.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

coeffs_matrix = loaded_dict["coeffs_matrix"]
K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)


# design siren NN for learning
characteristic_length = 64
synthesizer_nn = MLP(name="regressor_synthesizer",
                    input_size=3,
                    output_size=1,
                    hidden_layers=[characteristic_length] * 6,
                    activation_settings={"type":"sin",
                                         "prediction_gain":30,
                                         "initialization_gain":1.0})

latent_size = characteristic_length
modulator_nn = MLP(name="modulator_nn",
                   input_size=latent_size,
                   use_bias=False) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
num_epochs = 5000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.adam(1e-5))

# create fol
fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=identity_control,
                                                loss_function=reg_loss,
                                                flax_neural_network=hyper_network,
                                                main_loop_optax_optimizer=main_loop_transform,
                                                latent_step_size=1e-2,
                                                num_latent_iterations=3)

fol.Initialize()

train_start_id = 0
train_end_id = 80
test_start_id = 80
test_end_id = 100

fol.Train(train_set=(K_matrix[train_start_id:train_end_id,:],K_matrix[train_start_id:train_end_id,:]),
          test_set=(K_matrix[test_start_id:test_end_id,:],K_matrix[test_start_id:test_end_id,:]),
          test_frequency=100,batch_size=8,
          convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
          train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":1000},
          working_directory=case_dir)


# load the best model
fol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

test_ids = jnp.arange(0,50,5)
predictions = fol.Predict(K_matrix[test_ids,:])

for index,eval_id in enumerate(test_ids):
    predicted = predictions[index].reshape(-1)
    ground_truth = K_matrix[eval_id].reshape(-1)
    abs_err = abs(predicted-ground_truth)
    plot_mesh_vec_data(1,[predicted,ground_truth,abs_err],
                        ["predicted","ground_truth","abs_error"],
                        fig_title="",
                        file_name=os.path.join(case_dir,f"test_{eval_id}.png"))

