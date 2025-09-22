import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
#import jax
#jax.config.update("jax_platform_name","cpu")
import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.deep_neural_networks.nns import HyperNetwork,MLP
import pickle
import optax
from flax import nnx
from mechanical2d_utilities import *
from fol.tools.decoration_functions import *

### Script's goal:
####### the following script is to 
####### first load K_matrix from either a pkl file or txt file in base 
####### resolution, then train or load a pre-trained network to
####### save the output of FE and iFOL solver for Displacement/Stresses 
####### in a dictionary to be used in hybrid solver to solve for corresponding ZSSR 
####### resolution. It uses function dump_output to create plots and .json file storing
####### all network hyper-parameters.


def main(ifol_num_epochs=10,solve_FE=False,solve_FE_hybrid=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "ifol_output_mechanical_2d"
    case_dir = os.path.join('.', working_directory_name)
    # create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup
    model_settings = {"L":1,"N":81,
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

    identity_control = IdentityControl('identity_control', control_settings={}, fe_mesh=fe_mesh)
    identity_control.Initialize()
    
    
    #### load a binary file:

    # file_path = "sample_info_dict.pkl"
    # sample_info_dict = {}
    # with open(file_path, 'rb') as f:
    #     sample_info_dict = pickle.load(f)
    # fol_info(f"The file {file_path} loaded and stored in the variable sample_info_dict!")
    
    # # create K_matrix from saved topologies dictionary
    # K_matrix_all = []
    # for sample_type, sample_data in sample_info_dict.items():
    #     K_matrix_dict = sample_data.get("K_matrix",{})
    #     for axis_value , K_matrix_at_axis in K_matrix_dict.items():
    #         K_matrix_all.append(K_matrix_at_axis)

    # K_matrix = np.array(K_matrix_all)
    # fol_info(f"K_matrix shape loaded from {file_path}: {K_matrix.shape}")

    #### load a txt file:
    fourier_file = os.path.join(case_dir,f"ifol_fourier_test_samples_K_matrix_res_{model_settings['N']}.txt")
    voronoi_file = os.path.join(case_dir,f"ifol_voronoi_test_samples_K_matrix_res_{model_settings['N']}.txt")
    tpms_settings = {"phi_x": 0., "phi_y": 0., "phi_z": 0., "max": 1., "min": 0.02, "section_axis_value": 0.8,
                     "constant": 0., "threshold": 0.5, "coefficients":(2.,2.,2.)}
    K_matrix = create_tpms_schwarz_P(fe_mesh=fe_mesh,tpms_settings=tpms_settings).reshape(-1,1).T

    # tpms_file = os.path.join(case_dir,f"ifol_tpms_test_samples_K_matrix_res_{model_settings['N']}.txt")
    

    # K_matrix_fourier = np.loadtxt(fourier_file)
    # K_matrix_voronoi = np.loadtxt(voronoi_file)
    # K_matrix_tpms = np.loadtxt(tpms_file)

    # K_matrix_1 = np.row_stack((K_matrix_fourier,K_matrix_voronoi))
    # K_matrix = np.row_stack((K_matrix_1, K_matrix_tpms))


    # now we need to create, initialize and train fol
    ifol_settings_dict = {
        "characteristic_length": 64,
        "synthesizer_depth": 4,
        "activation_settings":{"type":"sin",
                                "prediction_gain":30,
                                "initialization_gain":1.0},
        "skip_connections_settings": {"active":False,"frequency":1},
        "latent_size":  8*64,
        "modulator_bias": False,
        "main_loop_transform": 1e-5,
        "latent_step_optimizer": 1e-4,
        "ifol_nn_latent_step_size": 1e-2
    }
    # design synthesizer & modulator NN for hypernetwork
    # characteristic_length = model_settings["N"]
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
    ifol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_fol",control=identity_control,
                                                            loss_function=mechanical_loss_2d,
                                                            flax_neural_network=hyper_network,
                                                            main_loop_optax_optimizer=main_loop_transform,
                                                            latent_step_optax_optimizer=latent_step_optimizer,
                                                            latent_step_size=ifol_settings_dict["ifol_nn_latent_step_size"],
                                                            num_latent_iterations=3)
    ifol.Initialize()

    otf_id = 0
    train_set_otf = K_matrix[otf_id,:].reshape(-1,1).T     # for On The Fly training

    train_start_id = 0
    train_end_id = 8
    train_set_pr = K_matrix[train_start_id,train_end_id]     # for parametric training

    test_start_id = 8
    test_end_id = 10
    test_set_pr = K_matrix[test_start_id,test_end_id]

    # OTF or Parametric 
    parametric_learning = False
    if parametric_learning:
        train_set = train_set_pr
        test_set = test_set_pr
        tests = range(test_start_id,test_start_id+5)
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
    
    solve_ifol = True
    if solve_ifol:
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

        print("K matrix: ", K_matrix.shape)
        if parametric_learning:
            iFOL_UVW = np.array(ifol.Predict(K_matrix))
        else:
            iFOL_UVW = np.array(ifol.Predict(K_matrix[otf_id].reshape(-1,1).T))
    U_dict = {}
    for eval_id in tests:
        if solve_ifol:
            ifol_uvw = np.array(iFOL_UVW[eval_id,:].reshape(-1,1).T)
            # ifol_stress = get_stress(loss_function=mechanical_loss_2d, fe_mesh=fe_mesh,
            #                               disp_field_vec=jnp.array(ifol_uvw), K_matrix=jnp.array(K_matrix[eval_id,:]))
            
            fe_mesh[f'iFOL_U_{eval_id}'] = ifol_uvw.reshape((fe_mesh.GetNumberOfNodes(), 2))
            fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(),1))
            # fe_mesh[f"iFOL_stress_{eval_id}"] = ifol_stress.reshape((fe_mesh.GetNumberOfNodes(), 3))
            # np.savetxt("ifol_solution.txt",ifol_uvw)

        # solve FE here
        if solve_FE:
            fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                        "nonlinear_solver_settings":{"rel_tol":1e-7,"abs_tol":1e-7,
                                                        "maxiter":8,"load_incr":51}}
            nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_2d,fe_setting)
            nonlin_fe_solver.Initialize()
            FE_UVW = np.array(nonlin_fe_solver.Solve(K_matrix[eval_id,:],np.zeros(2*fe_mesh.GetNumberOfNodes())))  

            fe_mesh[f'FE_U_{eval_id}'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))
            fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(),1))
            # fe_stress = get_stress(loss_function=mechanical_loss_2d, fe_mesh=fe_mesh,
            #                           disp_field_vec=jnp.array(FE_UVW), K_matrix=jnp.array(K_matrix[eval_id,:]))
            if solve_ifol:
                abs_err = abs(FE_UVW.reshape(-1,1) - ifol_uvw.reshape(-1,1))
                # abs_stress_err = abs(fe_stress.reshape(-1,1) - ifol_stress.reshape(-1,1))
                
                fe_mesh[f"abs_U_error_{eval_id}"] = abs_err.reshape((fe_mesh.GetNumberOfNodes(), 2))
                # fe_mesh[f"FE_stress_{eval_id}"] = fe_stress.reshape((fe_mesh.GetNumberOfNodes(), 3))
                # fe_mesh[f"abs_stress_error_{eval_id}"] = abs_stress_err.reshape((fe_mesh.GetNumberOfNodes(), 3))

                U_dict[f'U_iFOL_{eval_id}'] = ifol_uvw
                # U_dict[f'Stress_iFOL_{eval_id}'] = ifol_stress
                U_dict[f"abs_error_{eval_id}"] = abs_err
                # U_dict[f'abs_stress_error_{eval_id}'] = abs_stress_err

            
            U_dict[f'K_matrix_{eval_id}'] = K_matrix[eval_id,:]
            U_dict[f'U_FE_{eval_id}'] = FE_UVW
            # U_dict[f'Stress_FE_{eval_id}'] = fe_stress

            # # save solution for base resolution as fields in a pkl file.  
            # with open(f"U_base_res_{model_settings['N']}_bc_{model_settings['Ux_right']}.pkl" , 'wb') as f:
            #     pickle.dump(U_dict,f)

        if solve_FE_hybrid:
            # hybrid solver
            fol_info("solve fe hybrid in one load step")
            hfe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                        "nonlinear_solver_settings":{"rel_tol":1e-8,"abs_tol":1e-8,
                                                        "maxiter":10,"load_incr":1}}
            hybrid_nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("hybrid_nonlin_fe_solver",mechanical_loss_2d,hfe_setting)
            hybrid_nonlin_fe_solver.Initialize()
            try:    
                FE_H_UVW = np.array(hybrid_nonlin_fe_solver.Solve(K_matrix[eval_id,:],ifol_uvw.reshape(2*fe_mesh.GetNumberOfNodes())))  
            except:
                ValueError("res_norm contains nan values!")
                FE_H_UVW = np.zeros(2*fe_mesh.GetNumberOfNodes())

            fe_mesh[f'HFE_U_{eval_id}'] = FE_H_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))
            fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(),1))


            # plot_settings = {"topology_field":K_matrix[eval_id,:], "ifol_sol_field":ifol_uvw, "fe_sol_field":FE_UVW, 
            #                     "hfe_sol_field":FE_H_UVW, "err_sol_field":abs_err, 
            #                     "file_name":f"plot_ifol_fe_{eval_id}_res_{model_settings['N']}"}
            
            # dump_output(ifol_model_settings=ifol_settings_dict,
            #     train_settings_dict=train_settings_dict,
            #     model_settings=model_settings,
            #     material_dict=material_dict,
            #     plotter= plot_iFOL_FE_HFE,
            #     plot_settings=plot_settings,
            #     file_name=f"output_{eval_id}_res_{model_settings['N']}",case_dir=case_dir,file_format='json')


    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)



if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 3000
    solve_FE = True
    solve_FE_hybrid = False
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
        elif arg.startswith("solve_FE_hybrid="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                solve_FE_hybrid = value.lower() == 'true'
            else:
                print("solve_FE_hybrid should be True or False.")
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
    main(ifol_num_epochs,solve_FE,solve_FE_hybrid,clean_dir)
