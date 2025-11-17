import sys
import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import pickle,optax
import numpy as np
from fol.tools.usefull_functions import *
from fol.tools.decoration_functions import *
from fol.tools.logging_functions import Logger
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.loss_functions.transient_thermal import TransientThermalLoss2DQuad
from thermal_usefull_functions import *
import jax
jax.config.update("jax_default_matmul_precision", "highest")

# directory & save handling
working_directory_name = 'meta_implicit_pl_transient_nonlinear_thermal_2D'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":51,
                "T_left":1.0,"T_right":0.0}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()

# create some random fields for training
create_random_fields = True
if create_random_fields:
    train_temperature_fields = generate_random_smooth_patterns(model_settings["L"],model_settings["N"],num_samples=9000)
    train_heterogeneity_field = generate_morph_pattern(model_settings["N"]).reshape(1,-1) 
    train_data_dict = {"temperatures":train_temperature_fields,"heterogeneity":train_heterogeneity_field}
    with open(f'train_data_dict.pkl', 'wb') as f:
        pickle.dump(train_data_dict,f)
else:
    with open(f'train_data_dict.pkl', 'rb') as f:
        train_data_dict = pickle.load(f)
    
    train_temperature_fields = train_data_dict["temperatures"]
    train_heterogeneity_field = train_data_dict["heterogeneity"]

    fol_info(f"train temperature field {train_temperature_fields.shape} is imported !")
    fol_info(f"train heterogeneity field {train_heterogeneity_field.shape} is imported !")
    

# create fe-based loss function
bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings["T_right"]}}
material_dict = {"rho":1.0,"cp":1.0,"k0":train_heterogeneity_field.flatten(),"beta":1.5,"c":1.0}
time_integration_dict = {"method":"implicit-euler","time_step":0.005}
transient_thermal_loss_2d = TransientThermalLoss2DQuad("thermal_transient_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "num_gp":2,
                                                                            "material_dict":material_dict,
                                                                            "time_integration_dict":time_integration_dict},
                                                                            fe_mesh=fe_mesh)
transient_thermal_loss_2d.Initialize()

identity_control = IdentityControl("ident_control",num_vars=fe_mesh.GetNumberOfNodes())
identity_control.Initialize()

# design synthesizer & modulator NN for hypernetwork
characteristic_length = 256
synthesizer_nn = MLP(name="synthesizer_nn",
                     input_size=3,
                     output_size=1,
                     hidden_layers=[characteristic_length] * 6,
                     activation_settings={"type":"sin",
                                          "prediction_gain":30,
                                          "initialization_gain":1.0},
                     skip_connections_settings={"active":False,"frequency":1})

latent_size = characteristic_length
modulator_nn = MLP(name="modulator_nn",
                   input_size=latent_size,
                   use_bias=False) 

hyper_network = HyperNetwork(name="hyper_nn",
                             modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                             coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

# create fol optax-based optimizer
num_epochs = 10000
learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))

# create ifol
ifol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=identity_control,
                                            loss_function=transient_thermal_loss_2d,
                                            flax_neural_network=hyper_network,
                                            main_loop_optax_optimizer=main_loop_transform,
                                            latent_step_size=1e-2,
                                            num_latent_iterations=3)
ifol.Initialize()

train_start_id = 0
train_end_id = 6000

ifol.Train(train_set=(train_temperature_fields[train_start_id:train_end_id,:],),
            batch_size=120,
            convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
            plot_settings={"plot_save_rate":100},
            working_directory=case_dir)

ifol.RestoreState(restore_state_directory=case_dir+"/flax_final_state")

test_temperature_fields = generate_random_smooth_patterns_evaluation(model_settings["L"],model_settings["N"])

# time ineference setup
initial_solution_id = 0 
initial_solution = test_temperature_fields[initial_solution_id]
num_time_steps = 50

# predict dynamics with ifol
T_ifols = jnp.squeeze(ifol.PredictDynamics(initial_solution,num_time_steps))

# predict dynamics with FE
fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu"},
                "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                            "maxiter":10,"load_incr":1}}
nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",transient_thermal_loss_2d,fe_setting)
nonlin_fe_solver.Initialize()

T_fe_current = initial_solution.flatten()
T_fes = jnp.array(T_fe_current)
for _ in range(0,num_time_steps):
    T_fe_next = np.array(nonlin_fe_solver.Solve(T_fe_current,T_fe_current))
    T_fe_current = T_fe_next
    T_fes = jnp.vstack((T_fes,T_fe_next.flatten()))

# save the solutions
for time_index, (T_ifol,T_fe) in enumerate(zip(T_ifols,T_fes)):
    fe_mesh[f"T_ifol_{time_index}"] = np.array(T_ifol).reshape((fe_mesh.GetNumberOfNodes(), 1))
    fe_mesh[f"T_fe_{time_index}"] = np.array(T_fe).reshape((fe_mesh.GetNumberOfNodes(), 1))

fe_mesh["k0"] = np.array(train_heterogeneity_field.flatten()).reshape((fe_mesh.GetNumberOfNodes(), 1))


absolute_error = np.abs(T_ifols- T_fes)
time_list = [0,1,4,9,19,24,49]

plot_mesh_vec_data_thermal_row(1,[test_temperature_fields[initial_solution_id,:]],
                   [""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"initial_condition.png"))

plot_mesh_quad(fe_mesh.GetNodesCoordinates()[:,:-1],
               fe_mesh.GetElementsNodes("quad"),
               background=train_heterogeneity_field.reshape((model_settings["N"],model_settings["N"]))[::-1], 
               filename=os.path.join(case_dir,"FE_mesh_hetero_info.png"),show=False)

plot_mesh_vec_data_thermal_row(1,[T_ifols[time_list[0],:],T_ifols[time_list[1],:],
                                  T_ifols[time_list[2],:],T_ifols[time_list[3],:],
                                  T_ifols[time_list[4],:],T_ifols[time_list[5],:],
                                  T_ifols[time_list[6],:]],
                   ["","","","","","",""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"test2_FOL_summary.png"))
plot_mesh_vec_data_thermal_row(1,[T_fes[time_list[0],:],T_fes[time_list[1],:],
                                  T_fes[time_list[2],:],T_fes[time_list[3],:],
                                  T_fes[time_list[4],:],T_fes[time_list[5],:],
                                  T_fes[time_list[6],:]],
                   ["","","","","","",""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"test2_FE_summary.png"))

plot_mesh_vec_data_thermal_row(1,[absolute_error[time_list[0],:],absolute_error[time_list[1],:],
                                  absolute_error[time_list[2],:],absolute_error[time_list[3],:],
                                  absolute_error[time_list[4],:],absolute_error[time_list[5],:],
                                  absolute_error[time_list[6],:]],
                   ["","","","","","",""],
                   fig_title="",cmap = "jet",
                   file_name=os.path.join(case_dir,"test2_Error_summary.png"))


fe_mesh.Finalize(export_dir=case_dir)