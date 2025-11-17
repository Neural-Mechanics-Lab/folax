import sys
import os
import optax
import numpy as np
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.loss_functions.phase_field import AllenCahnLoss2DTri
from fol.controls.identity_control import IdentityControl
from allen_cahn_usefull_functions import *
import pickle
import jax
jax.config.update("jax_default_matmul_precision", "highest")

# directory & save handling
working_directory_name = 'transient_allen_cahn_2D'
case_dir = os.path.join('.', working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

fe_mesh = Mesh("fol_io","Li_battery_particle_bumps.med",'../../meshes/')

bc_dict = {"Phi":{}}
material_dict = {"rho":1.0,"cp":1.0,"dt":0.001,"epsilon":0.2}
phasefield_loss_2d = AllenCahnLoss2DTri("phasefield_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                            "material_dict":material_dict},
                                                                            fe_mesh=fe_mesh)
fe_mesh.Initialize()
phasefield_loss_2d.Initialize()

# create identity control
identity_control = IdentityControl("ident_control",num_vars=phasefield_loss_2d.GetNumberOfUnknowns())
identity_control.Initialize()

# generate some randome spatial fields
generate_new_samples = True
if generate_new_samples:
    sample_matrix = generate_random_smooth_patterns(fe_mesh.GetNodesCoordinates(), num_samples=10000)
    with open(f'sample_matrix.pkl', 'wb') as f:
        pickle.dump({"sample_matrix":sample_matrix},f)
else:
    with open(f'sample_matrix.pkl', 'rb') as f:
        sample_matrix = pickle.load(f)["sample_matrix"]

characteristic_length = 256
synthesizer_nn = MLP(name="synthesizer_nn",
                     input_size=2,
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
                                            loss_function=phasefield_loss_2d,
                                            flax_neural_network=hyper_network,
                                            main_loop_optax_optimizer=main_loop_transform,
                                            latent_step_size=1e-2,
                                            num_latent_iterations=3)
ifol.Initialize()

train_start_id = 0
train_end_id = 8000
ifol.Train(train_set=(sample_matrix[train_start_id:train_end_id,:],),
          batch_size=100,
          convergence_settings={"num_epochs":num_epochs,
                                "relative_error":1e-100,
                                "absolute_error":1e-100},
          working_directory=case_dir)

# load the best model
ifol.RestoreState(restore_state_directory=case_dir+"/flax_final_state")

# time ineference setup
test_samples = generate_fixed_gaussian_basis_field(fe_mesh.GetNodesCoordinates(), num_samples=5, length_scale=0.15)
initial_solution_id = 0 
initial_solution = test_samples[initial_solution_id]
num_time_steps = 10

# predict dynamics with ifol
phi_ifols = jnp.squeeze(ifol.PredictDynamics(initial_solution,num_time_steps))

# predict dynamics with FE
fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                                "maxiter":1000,"pre-conditioner":"ilu"},
                "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                            "maxiter":10,"load_incr":1}}
nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",phasefield_loss_2d,fe_setting)
nonlin_fe_solver.Initialize()

phi_fe_current = initial_solution.flatten()
phi_fes = jnp.array(phi_fe_current)
for _ in range(0,num_time_steps):
    phi_fe_next = np.array(nonlin_fe_solver.Solve(phi_fe_current,phi_fe_current))
    phi_fe_current = phi_fe_next
    phi_fes = jnp.vstack((phi_fes,phi_fe_next.flatten()))

# save the solutions
for time_index, (phi_ifol,phi_fe) in enumerate(zip(phi_ifols,phi_fes)):
    fe_mesh[f"phi_ifol_{time_index}"] = np.array(phi_ifol).reshape((fe_mesh.GetNumberOfNodes(), 1))
    fe_mesh[f"phi_fe_{time_index}"] = np.array(phi_fe).reshape((fe_mesh.GetNumberOfNodes(), 1))

absolute_error = np.abs(phi_fes-phi_ifols)

time_list = [0,1,4,9]

plot_triangulated(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [initial_solution],
                  filename=os.path.join(case_dir,f"test3_init_phi.png"),value_range=(-1,1),row=True)
plot_triangulated(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [phi_ifols[time_list[0],:],phi_ifols[time_list[1],:],
                   phi_ifols[time_list[2],:],phi_ifols[time_list[3],:]],
                  filename=os.path.join(case_dir,f"test3_FOL_summary.png"),value_range=(-1,1),row=True)
plot_triangulated(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [phi_fes[time_list[0],:],phi_fes[time_list[1],:],
                   phi_fes[time_list[2],:],phi_fes[time_list[3],:]],
                  filename=os.path.join(case_dir,f"test3_FE_summary.png"),value_range=(-1,1),row=True)
plot_triangulated(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                  [absolute_error[time_list[0],:],absolute_error[time_list[1],:],
                   absolute_error[time_list[2],:],absolute_error[time_list[3],:]],
                  filename=os.path.join(case_dir,f"test3_Error_summary.png"),value_range=None,row=True)
plot_mesh_tri(fe_mesh.GetNodesCoordinates(),fe_mesh.GetElementsNodes("triangle"),
                            filename=os.path.join(case_dir,f'test_FE_mesh_particle.png'))

fe_mesh.Finalize(export_dir=case_dir)
