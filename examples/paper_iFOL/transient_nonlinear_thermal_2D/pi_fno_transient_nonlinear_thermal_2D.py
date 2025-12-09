import sys
import os
import pickle,optax
import numpy as np
from fol.tools.usefull_functions import *
from fol.tools.decoration_functions import *
from fol.tools.logging_functions import Logger
from fol.controls.identity_control import IdentityControl
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.loss_functions.transient_thermal import TransientThermalLoss2DQuad
from thermal_usefull_functions import *
from fol.deep_neural_networks.fourier_neural_operator_networks import FourierNeuralOperator2D
from fol.deep_neural_networks.fourier_parametric_operator_learning import PhysicsInformedFourierParametricOperatorLearning

from fol.tools.decoration_functions import *
from flax.nnx import bridge
import pickle
from flax import nnx
import jax
jax.config.update("jax_default_matmul_precision", "highest")

# directory & save handling
working_directory_name = 'pi_fno_transient_nonlinear_thermal_2D'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

# problem setup
model_settings = {"L":1,"N":22,
                "T_left":1.0,"T_right":0.0}

# creation of the model
fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
fe_mesh.Initialize()

# create some random fields for training
create_random_fields = True
if create_random_fields:
    train_temperature_fields = generate_random_smooth_patterns(model_settings["L"],model_settings["N"],num_samples=1000)
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
  
train_heterogeneity_field = generate_morph_pattern(model_settings["N"]).reshape(1,-1) 
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

def merge_state(dst: nnx.State, src: nnx.State):
    for k, v in src.items():
        if isinstance(v, nnx.State):
            merge_state(dst[k], v)
        else:
            dst[k] = v

fno_model = bridge.ToNNX(FourierNeuralOperator2D(modes1=10,
                                                modes2=10,
                                                width=10,
                                                depth=3,
                                                channels_last_proj=128,
                                                out_channels=1,
                                                output_scale=1.0),rngs=nnx.Rngs(0)).lazy_init(train_heterogeneity_field[0:1].reshape(1,model_settings["N"],model_settings["N"],1)) 

# get total number of fno params
params = nnx.state(fno_model, nnx.Param)
total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
print(f"total number of fno network param:{total_params}")

num_epochs = 1000
learning_rate_scheduler = optax.linear_schedule(init_value=5e-3, end_value=5e-4, transition_steps=num_epochs)
optimizer = optax.chain(optax.adam(learning_rate_scheduler))

# create fol
pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(name="pi_fno_pr_learning",
                                                                        control=identity_control,
                                                                        loss_function=transient_thermal_loss_2d,
                                                                        flax_neural_network=fno_model,
                                                                        optax_optimizer=optimizer)

pi_fno_pr_learning.Initialize()

train_start_id = 0
train_end_id = 900
print(train_temperature_fields.shape)
pi_fno_pr_learning.Train(train_set=(train_temperature_fields[train_start_id:train_end_id,:],),
                         test_set=(train_temperature_fields[train_end_id:,:],),
            batch_size=50,
            convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
            plot_settings={"plot_save_rate":100},
            working_directory=case_dir)

pi_fno_pr_learning.RestoreState(restore_state_directory=case_dir+"/flax_final_state")


# time ineference setup
initial_solution_id = 900
initial_solution = train_temperature_fields[initial_solution_id]
num_time_steps = 50

# predict dynamics with ifol
T_ifols = pi_fno_pr_learning.PredictDynamics(initial_solution,num_time_steps)

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

plot_mesh_vec_data_thermal_row(1,[train_temperature_fields[initial_solution_id,:]],
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