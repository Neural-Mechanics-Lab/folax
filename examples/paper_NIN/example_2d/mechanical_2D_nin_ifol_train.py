import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import jax
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.identity_control import IdentityControl
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
import optax
from flax import nnx
from mechanical2d_utilities import *
from fol.tools.decoration_functions import *


def main(ifol_num_epochs=10,solve_FE=False,solve_NiN=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "nn_output_NiN"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup
    model_settings = {"L":1,"N":41,
                    "Ux_left":0.0,"Ux_right":0.5,
                    "Uy_left":0.0,"Uy_right":0.5}

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
    fe_mesh.Initialize()

    # create fe-based loss function
    bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
               "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}
    
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}

    mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                        "num_gp":2,
                                                                                        "material_dict":material_dict},
                                                        fe_mesh=fe_mesh)

    mechanical_loss_2d.Initialize()

    identity_control = IdentityControl('identity_control', control_settings={},num_vars=fe_mesh.GetNumberOfNodes())
    identity_control.Initialize()
    
    
    # fourier control
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)
    fourier_control.Initialize()

    # create some random coefficients & K for training
    create_random_coefficients = True
    if create_random_coefficients:
        number_of_random_samples = 200
        coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
        export_dict = model_settings.copy()
        export_dict["coeffs_matrix"] = coeffs_matrix
        export_dict["x_freqs"] = fourier_control.x_freqs
        export_dict["y_freqs"] = fourier_control.y_freqs
        export_dict["z_freqs"] = fourier_control.z_freqs
        with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'wb') as f:
            pickle.dump(export_dict,f)
    else:
        with open(f'fourier_control_dict_N_{model_settings["N"]}.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        
        coeffs_matrix = loaded_dict["coeffs_matrix"]

    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)


    # now we need to create, initialize and train fol
    ifol_settings_dict = {
        "characteristic_length": 64,
        "synthesizer_depth": 4,
        "activation_settings":{"type":"sin",
                                "prediction_gain":30,
                                "initialization_gain":1.0},
        "skip_connections_settings": {"active":False,"frequency":1},
        "latent_size":  64,
        "modulator_bias": False,
        "main_loop_transform": 1e-5,
        "latent_step_optimizer": 1e-4,
        "ifol_nn_latent_step_size": 1e-2
    }

    # design synthesizer & modulator NN for hypernetwork
    characteristic_length = ifol_settings_dict["characteristic_length"]
    synthesizer_nn = MLP(name="synthesizer_nn",
                        input_size=3,
                        output_size=2,
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
                                                            loss_function=mechanical_loss_2d,
                                                            flax_neural_network=hyper_network,
                                                            main_loop_optax_optimizer=main_loop_transform,
                                                            latent_step_optax_optimizer=latent_step_optimizer,
                                                            latent_step_size=ifol_settings_dict["ifol_nn_latent_step_size"],
                                                            num_latent_iterations=3)
    ifol.Initialize()

    otf_id = 0
    train_set_otf = coeffs_matrix[otf_id,:].reshape(-1,1).T     # for On The Fly training

    train_start_id = 0
    train_end_id = 8
    train_set_pr = coeffs_matrix[train_start_id:train_end_id,:]     # for parametric training

    test_start_id = 8
    test_end_id = 10
    test_set_pr = coeffs_matrix[test_start_id:test_end_id,:]

    # OTF or Parametric 
    parametric_learning = False
    if parametric_learning:
        train_set = train_set_pr
        test_set = test_set_pr
        tests = range(test_start_id,test_end_id)
    else:
        train_set = train_set_otf   
        test_set = train_set
        tests = [otf_id]
    #here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
    train_settings_dict = {"batch_size": 1,
                            "num_epoch":ifol_num_epochs,
                            "parametric_learning": parametric_learning,
                            "OTF_id": otf_id,
                            "train_start_id": train_start_id,
                            "train_end_id": train_end_id,
                            "test_start_id": test_start_id,
                            "test_end_id": test_end_id}
    
    ifol.Train(train_set=(train_set,),
                test_set=(test_set,),
                test_frequency=100,
                batch_size=train_settings_dict["batch_size"],
                convergence_settings={"num_epochs":train_settings_dict["num_epoch"],"relative_error":1e-100,"absolute_error":1e-100},
                plot_settings={"plot_save_rate":100},
                train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
                working_directory=case_dir)

    # load teh best model
    ifol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

    U_dict = {}
    for eval_id in tests:
        ifol_uvw = np.array(ifol.Predict(coeffs_matrix[otf_id].reshape(-1,1).T))
        fe_mesh[f'iFOL_U_{eval_id}'] = ifol_uvw.reshape((fe_mesh.GetNumberOfNodes(), 2))
        fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(),1))

        # solve FE here
        if solve_FE:
            fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                        "nonlinear_solver_settings":{"rel_tol":1e-7,"abs_tol":1e-7,
                                                        "maxiter":8,"load_incr":31}}
            nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_2d,fe_setting)
            nonlin_fe_solver.Initialize()
            FE_UVW = np.array(nonlin_fe_solver.Solve(K_matrix[eval_id,:],np.zeros(2*fe_mesh.GetNumberOfNodes())))  

            fe_mesh[f'FE_U_{eval_id}'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))

            abs_err = abs(FE_UVW.reshape(-1,1) - ifol_uvw.reshape(-1,1))
            fe_mesh[f"abs_U_error_{eval_id}"] = abs_err.reshape((fe_mesh.GetNumberOfNodes(), 2))

            U_dict[f'U_iFOL_{eval_id}'] = ifol_uvw
            U_dict[f"abs_error_{eval_id}"] = abs_err
            U_dict[f'K_matrix_{eval_id}'] = K_matrix[eval_id,:]
            U_dict[f'U_FE_{eval_id}'] = FE_UVW

            # # save solution for base resolution as fields in a pkl file.  
            # with open(f"U_base_res_{model_settings['N']}_bc_{model_settings['Ux_right']}.pkl" , 'wb') as f:
            #     pickle.dump(U_dict,f)

        if solve_NiN:
            # hybrid solver
            fol_info("solve fe hybrid in one load step")
            nin_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                        "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                                        "maxiter":10,"load_incr":1}}
            nin_nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nin_nonlin_fe_solver",mechanical_loss_2d,nin_setting)
            nin_nonlin_fe_solver.Initialize()
            try:    
                NiN_UVW = np.array(nin_nonlin_fe_solver.Solve(K_matrix[eval_id,:],ifol_uvw.reshape(2*fe_mesh.GetNumberOfNodes())))  
            except Exception as e:
                print(f"Error occured {type(e).__name__}: e")
                NiN_UVW = np.zeros(2*fe_mesh.GetNumberOfNodes())

            fe_mesh[f'NiN_U_{eval_id}'] = NiN_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))
            fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(),1))

            plot_iFOL_HFE(topology_field=K_matrix[eval_id,:], ifol_sol_field=ifol_uvw.reshape(2*fe_mesh.GetNumberOfNodes()), hfe_sol_field=NiN_UVW,
                     err_sol_field=abs_err, file_name=os.path.join(case_dir,'plots')+f"/ifol_fe-nin_error_{eval_id}",
                     fig_titles=['Elasticity Morph.','iFOL','FE-NIN','iFOL-FE Abs Diff.'])


    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)



if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 5000
    solve_FE = True
    solve_NiN = True
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
        elif arg.startswith("solve_NiN="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                solve_NiN = value.lower() == 'true'
            else:
                print("solve_NiN should be True or False.")
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
    main(ifol_num_epochs,solve_FE,solve_NiN,clean_dir)
