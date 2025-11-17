# import necessaries 
import sys
import os
import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import numpy as np
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle,optax
from fol.tools.decoration_functions import *
import timeit
import numpy as np
import statistics
import jax
from fol.loss_functions.regression_loss import RegressionLoss


# directory & save handling
working_directory_name = "fourier_box_3D_tetra"
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# create mesh
fe_mesh = Mesh("box_3D","box_3D_coarse.med",'../meshes/')

# create regression loss
reg_loss = RegressionLoss("reg_loss",loss_settings={"nodal_unknows":["K"]},fe_mesh=fe_mesh)

# fourier control
fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                            "beta":20,"min":1e-2,"max":1}
fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

fe_mesh.Initialize()
reg_loss.Initialize()
fourier_control.Initialize()
# create identity control
identity_control = IdentityControl("ident_control",num_vars=fe_mesh.GetNumberOfNodes())
identity_control.Initialize()

# create/load some random coefficients & K for training
create_random_coefficients = False
if create_random_coefficients:
    number_of_samples = 10000
    coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_samples)
    export_dict = {}
    export_dict["coeffs_matrix"] = coeffs_matrix
    export_dict["x_freqs"] = fourier_control.x_freqs
    export_dict["y_freqs"] = fourier_control.y_freqs
    export_dict["z_freqs"] = fourier_control.z_freqs
    with open(f'fourier_3D_control_dict_10K.pkl', 'wb') as f:
        pickle.dump(export_dict,f)
else:
    with open(f'fourier_3D_control_dict_10K.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]

n_repeat = 5   # How many times to repeat the timing
n_number = 5    # How many times to run the function in each repeat

def test_perf_function(function_name,test_function,test_inputs): 
    jit_time = timeit.repeat(lambda: test_function(test_inputs).block_until_ready,repeat=1, number=1)    
    times = timeit.repeat(lambda: test_function(test_inputs).block_until_ready,repeat=n_repeat, number=n_number)
    normalized_times = np.array(times) / n_number
    print(f"{function_name} statistics:")
    print(f"jit time: {jit_time[0]:.6f} sec")
    print(f"Per-run mean time: {statistics.mean(normalized_times):.6f} sec")
    print(f"Per-run std dev  : {statistics.stdev(normalized_times):.6f} sec")      

K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

# test_perf_function("ComputeBatchControlledVariables",fourier_control.ComputeBatchControlledVariables,coeffs_matrix)


# now create implicit parametric deep learning
# design synthesizer & modulator NN for hypernetwork
characteristic_length = 256
synthesizer_nn = MLP(name="synthesizer_nn",
                     input_size=3,
                     output_size=1,
                     hidden_layers=[characteristic_length] * 10,
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

key = jax.random.PRNGKey(0)
latents = jax.random.uniform(key, shape=(K_matrix.shape[0], latent_size), minval=0.0, maxval=1.0)
coords = fe_mesh.GetNodesCoordinates()
# preds = hyper_network(latents,coords)

# loss_values = reg_loss.ComputeBatchLoss(K_matrix,preds)

# print(loss_values.shape)
# print(jnp.sum(loss_values))

# exit()

# create fol optax-based optimizer
num_epochs = 2000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.adam(1e-5))
latent_step_optimizer = optax.chain(optax.adam(1e-1))



if True:
    ifol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",
                                                control=identity_control,
                                                loss_function=reg_loss,
                                                flax_neural_network=hyper_network,
                                                main_loop_optax_optimizer=main_loop_transform,
                                                latent_step_size=1e-2,
                                                num_latent_iterations=3)
else:

    ifol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=identity_control,
                                                    loss_function=reg_loss,
                                                    flax_neural_network=hyper_network,
                                                    main_loop_optax_optimizer=main_loop_transform,
                                                    latent_step_size=1e-2,
                                                    latent_step_optax_optimizer=latent_step_optimizer,
                                                    num_latent_iterations=3)


ifol.Initialize()

ifol.Train(train_set=(K_matrix,K_matrix),
        test_frequency=10,
        batch_size=128,
        convergence_settings={"num_epochs":2000,"relative_error":1e-100,"absolute_error":1e-100},
        plot_settings={"save_frequency":100},
        train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":1000,"state_directory":case_dir+"/flax_train_state"},
        test_checkpoint_settings={"least_loss_checkpointing":False,"frequency":1000,"state_directory":case_dir+"/flax_test_state"},
        restore_nnx_state_settings={"restore":False,"state_directory":case_dir+"/flax_final_state"},
        working_directory=case_dir)

