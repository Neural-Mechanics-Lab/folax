import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..',)))
import optax
from flax import nnx
import jax
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.fourier_parametric_operator_learning import PhysicsInformedFourierParametricOperatorLearning
from fol.deep_neural_networks.fourier_neural_operator_networks import FourierNeuralOperator2D
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
import pickle, time
from flax.nnx import bridge

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = 'nn_output_mechanical_2D_neohooke_pi_fno'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup
    model_settings = {"L":1,"N":42,
                    "Ux_left":0.0,"Ux_right":0.1,
                    "Uy_left":0.0,"Uy_right":0.1}

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

    # create fe-based loss function
    bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
               "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}
    
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                              "num_gp":2,
                                                                              "material_dict":material_dict},
                                                                              fe_mesh=fe_mesh)

    # fourier control
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

    fe_mesh.Initialize()
    mechanical_loss_2d.Initialize()
    fourier_control.Initialize()

    # create some random coefficients & K for training
    create_random_coefficients = False
    if create_random_coefficients:
        number_of_random_samples = 200
        coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
        export_dict = model_settings.copy()
        export_dict["coeffs_matrix"] = coeffs_matrix
        export_dict["x_freqs"] = fourier_control.x_freqs
        export_dict["y_freqs"] = fourier_control.y_freqs
        export_dict["z_freqs"] = fourier_control.z_freqs
        with open(f'fourier_control_dict.pkl', 'wb') as f:
            pickle.dump(export_dict,f)
    else:
        with open(f'fourier_control_dict.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        
        coeffs_matrix = loaded_dict["coeffs_matrix"]

    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

    dofs = mechanical_loss_2d.GetDOFs()
    def merge_state(dst: nnx.State, src: nnx.State):
        for k, v in src.items():
            if isinstance(v, nnx.State):
                merge_state(dst[k], v)
            else:
                dst[k] = v

    fno_model = bridge.ToNNX(FourierNeuralOperator2D(modes1=6,
                                                    modes2=6,
                                                    width=8,
                                                    depth=4,
                                                    channels_last_proj=32,
                                                    out_channels=2,
                                                    output_scale=0.1),rngs=nnx.Rngs(0)).lazy_init(K_matrix[0:1].reshape(1,model_settings["N"],model_settings["N"],1)) 

    # replace RNG key by a dummy to allow checkpoint restoration later
    graph_def, state = nnx.split(fno_model)
    rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))
    merge_state(state, rngs_key)
    fno_model = nnx.merge(graph_def, state)

    # get total number of fno params
    params = nnx.state(fno_model, nnx.Param)
    total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print(f"total number of fno network param:{total_params}")

    num_epochs = 1000
    learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-5, transition_steps=num_epochs)
    optimizer = optax.chain(optax.adam(1e-3))

    # create fol
    pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(name="pi_fno_pr_learning",
                                                                            control=fourier_control,
                                                                            loss_function=mechanical_loss_2d,
                                                                            flax_neural_network=fno_model,
                                                                            optax_optimizer=optimizer)

    pi_fno_pr_learning.Initialize()

    otf_id = 0
    train_start_id = 0
    train_end_id = 10
    test_start_id = 180
    test_end_id = 182
    
    # Parametric learning or On The Fly learning
    parametric_learning = True
    if parametric_learning:
        train_set = coeffs_matrix[train_start_id:train_end_id,:]
        test_set = coeffs_matrix[test_start_id:test_end_id,:]
        eval_cases = range(test_start_id,test_end_id)
        batch_size = 5
    else:
        train_set = coeffs_matrix[otf_id,:].reshape(-1,1).T
        test_set = coeffs_matrix[otf_id,:].reshape(-1,1).T
        eval_cases = [otf_id]
        batch_size = 1

    #here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
    pi_fno_pr_learning.Train(train_set=(train_set,),
                            test_set=(test_set,),
                            test_frequency=100,
                            batch_size=batch_size,
                            convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                            plot_settings={"plot_save_rate":100},
                            train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
                            working_directory=case_dir)

    # load teh best model
    pi_fno_pr_learning.RestoreState(restore_state_directory=case_dir+"/flax_train_state")


    for eval_id in eval_cases:
        FNO_UV = np.array(pi_fno_pr_learning.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
        fe_mesh[f'U_FNO_{eval_id}'] = FNO_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

        # solve FE here
        fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                                    "maxiter":1000,"pre-conditioner":"ilu"},
                      "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                    "maxiter":8,"load_incr":21}}
        nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_2d,fe_setting)
        nonlin_fe_solver.Initialize()
        FE_UV = np.array(nonlin_fe_solver.Solve(K_matrix[eval_id],np.zeros(2*fe_mesh.GetNumberOfNodes())))  

        fe_mesh[f'U_FE_{eval_id}'] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

        absolute_error = abs(FNO_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
        fe_mesh[f'abs_error_{eval_id}'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 2))
        
        plot_mesh_vec_data(model_settings["L"], [K_matrix[eval_id,:],FNO_UV[::2],FE_UV[::2],absolute_error[::2]], 
                        subplot_titles= ['Heterogeneity', 'FNO_U', 'FE_U', "absolute_error"], fig_title=None, cmap='viridis',
                            block_bool=True, colour_bar=True, colour_bar_name=None,
                            X_axis_name=None, Y_axis_name=None, show=False, file_name=os.path.join(case_dir,f'plot_results_{eval_id}.png'))
    
    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    fol_num_epochs = 2000
    solve_FE = False
    clean_dir = False

    # Parse the command-line arguments
    args = sys.argv[1:]

    # Process the arguments if provided
    for arg in args:
        if arg.startswith("fol_num_epochs="):
            try:
                fol_num_epochs = int(arg.split("=")[1])
            except ValueError:
                print("fol_num_epochs should be an integer.")
                sys.exit(1)
        elif arg.startswith("solve_FE="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                solve_FE = value.lower() == 'true'
            else:
                print("solve_FE should be True or False.")
                sys.exit(1)
        elif arg.startswith("clean_dir="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                clean_dir = value.lower() == 'true'
            else:
                print("clean_dir should be True or False.")
                sys.exit(1)
        else:
            print("Usage: python mechanical_2D.py fol_num_epochs=10 solve_FE=False clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(fol_num_epochs, solve_FE,clean_dir)