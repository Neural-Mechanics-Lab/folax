import os,time,sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..')))
import numpy as np
import optax
from flax import nnx
import jax
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.thermal import ThermalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle
from utilities import *

def main(ifol_num_epochs=10,solve_FE=False,clean_dir=False):
    # cleaning & logging
    working_directory_name = 'nn_output_thermal_pi_ifol_2D'
    case_dir = os.path.join('.', working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,"fol_thermal_2D.log"))

    # create mesh_io
    # problem setup
    model_settings = {"L":1,"N":52,
                    "left":1.0,"right":0.1}

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

    # create fe-based loss function
    bc_dict = {"T":{"left":model_settings["left"],"right":model_settings["right"]}}

    thermal_loss_2d = ThermalLoss2DQuad("thermal_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,"loss_function_exponent":2,
                                                                          "beta":2,"c":4},
                                                                            fe_mesh=fe_mesh)

    # create Fourier parametrization/control
    x_freqs = np.array([3,5,7])
    y_freqs = np.array([2,4,7])
    z_freqs = np.array([0])
    fourier_control_settings = {"x_freqs":x_freqs,"y_freqs":y_freqs,"z_freqs":z_freqs,"beta":5,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

    fe_mesh.Initialize()
    thermal_loss_2d.Initialize()
    fourier_control.Initialize()

    # create some random coefficients & K for training
    create_random_coefficients = False
    if create_random_coefficients:
        number_of_random_samples = 10000
        coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
        export_dict = {}
        export_dict["coeffs_matrix"] = coeffs_matrix
        export_dict["x_freqs"] = x_freqs
        export_dict["y_freqs"] = y_freqs
        export_dict["z_freqs"] = z_freqs
        with open(f'fourier_control_dict_efol_paper.pkl', 'wb') as f:
            pickle.dump(export_dict,f)
    else:
        with open(os.path.join(os.path.dirname(__file__),f'fourier_control_dict_efol_paper.pkl'), 'rb') as f:
            loaded_dict = pickle.load(f)
        
        coeffs_matrix = loaded_dict["coeffs_matrix"]
        K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)


    # now save K matrix 
    export_Ks = False
    if export_Ks:
        for i in range(K_matrix.shape[0]):
            fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
        fe_mesh.Finalize(export_dir=case_dir)


    # now we need to create, initialize and train fol
    ifol_settings_dict = {
        "characteristic_length": 64,
        "synthesizer_depth": 6,
        "activation_settings":{"type":"sin",
                                "prediction_gain":10,
                                "initialization_gain":1.0},
        "skip_connections_settings": {"active":False,"frequency":1},
        "latent_size":  8*64,
        "modulator_bias": False,
        "main_loop_transform": 1e-6,
        "latent_step_optimizer": 1e-5,
        "ifol_nn_latent_step_size": 1e-4
    }

    # design synthesizer & modulator NN for hypernetwork
    characteristic_length = ifol_settings_dict["characteristic_length"]
    synthesizer_nn = MLP(name="synthesizer_nn",
                        input_size=3,
                        output_size=1,
                        hidden_layers=[characteristic_length] * ifol_settings_dict["synthesizer_depth"],
                        activation_settings=ifol_settings_dict["activation_settings"],
                        skip_connections_settings=ifol_settings_dict["skip_connections_settings"])

    latent_size = ifol_settings_dict["latent_size"]
    modulator_nn = MLP(name="modulator_nn",
                    input_size=latent_size,
                    use_bias=ifol_settings_dict["modulator_bias"]) 

    hyper_network = HyperNetwork(name="hyper_nn",
                                modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                                coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

    # create fol optax-based optimizer
    #learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
    main_loop_transform = optax.chain(optax.adam(ifol_settings_dict["main_loop_transform"]))
    latent_step_optimizer = optax.chain(optax.adam(ifol_settings_dict["latent_step_optimizer"]))

    # create fol
    ifol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_fol",control=fourier_control,
                                                            loss_function=thermal_loss_2d,
                                                            flax_neural_network=hyper_network,
                                                            main_loop_optax_optimizer=main_loop_transform,
                                                            latent_step_optax_optimizer=latent_step_optimizer,
                                                            latent_step_size=ifol_settings_dict["ifol_nn_latent_step_size"],
                                                            num_latent_iterations=3)
    ifol.Initialize()

    otf_id = 0
    train_set_otf = coeffs_matrix[otf_id,:].reshape(-1,1).T     # for On The Fly training

    train_start_id = 0
    train_end_id = 8000
    train_set_pr = coeffs_matrix[train_start_id:train_end_id,:]     # for parametric training

    test_start_id = 8000
    test_end_id = 9000
    test_set_pr = coeffs_matrix[test_start_id:test_end_id,:]

    # OTF or Parametric 
    parametric_learning = True
    if parametric_learning:
        train_set = train_set_pr
        test_set = test_set_pr
        tests = range(test_start_id,test_end_id)
    else:
        train_set = train_set_otf   
        test_set = train_set
        tests = [otf_id]
    #here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
    train_settings_dict = {"batch_size": 100,
                            "num_epoch":ifol_num_epochs,
                            "parametric_learning": parametric_learning,
                            "OTF_id": otf_id,
                            "train_start_id": train_start_id,
                            "train_end_id": train_end_id,
                            "test_start_id": test_start_id,
                            "test_end_id": test_end_id}
    
    # ifol.Train(train_set=(train_set,),
    #             test_set=(test_set,),
    #             test_frequency=100,
    #             batch_size=train_settings_dict["batch_size"],
    #             convergence_settings={"num_epochs":train_settings_dict["num_epoch"],"relative_error":1e-100,"absolute_error":1e-100},
    #             plot_settings={"plot_save_rate":100},
    #             train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
    #             working_directory=case_dir)

    # load teh best model
    ifol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

    iFOL_dict = {}
    iFOL = np.array(ifol.Predict(coeffs_matrix))
    iFOL_dict['iFOL_pred'] = iFOL
    with open(f'thermal_ifol_pred_dict.pkl', 'wb') as f:
        pickle.dump(iFOL_dict,f)

    for eval_id in [9000,9121,9236,9347,9486,9563,9617,9785,9863,9963,9988,9999]:

        iFOL_T = np.array(ifol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T)).reshape(-1)
        fe_mesh[f'T_iFOL_{eval_id}'] = iFOL_T.reshape((fe_mesh.GetNumberOfNodes(), 1))
        fe_mesh[f'K_{eval_id}'] = K_matrix[eval_id].reshape((fe_mesh.GetNumberOfNodes(), 1))

        # solve FE here
        if solve_FE: 
            fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                        "nonlinear_solver_settings":{"rel_tol":1e-6,"abs_tol":1e-6,
                                                        "maxiter":20,"load_incr":4}}
            nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",thermal_loss_2d,fe_setting)
            nonlin_fe_solver.Initialize()
            FE_T = np.array(nonlin_fe_solver.Solve(K_matrix[eval_id],np.zeros(fe_mesh.GetNumberOfNodes())))  
            fe_mesh[f'T_FE_{eval_id}'] = FE_T.reshape((fe_mesh.GetNumberOfNodes(), 1))

            absolute_error = abs(iFOL_T.reshape(-1,1)-FE_T.reshape(-1,1))
            relative_error = 100 * absolute_error/abs(FE_T.reshape(-1,1))
            fe_mesh[f'relative_error_{eval_id}'] = relative_error.reshape((fe_mesh.GetNumberOfNodes(), 1))
            fe_mesh[f'absolute_error_{eval_id}'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 1))

            plot_thermal_paper(vectors_list=[K_matrix[eval_id],iFOL_T,FE_T], file_name=case_dir+f"/thermal_2d_sample_{eval_id}")

    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 5000
    solve_FE = True
    clean_dir = False

    # Parse the command-line arguments
    args = sys.argv[1:]

    # Process the arguments if provided
    for arg in args:
        if arg.startswith("fol_num_epochs="):
            try:
                ifol_num_epochs = int(arg.split("=")[1])
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
            print("Usage: python thermal_fol.py fol_num_epochs=10 solve_FE=False clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(ifol_num_epochs, solve_FE,clean_dir)