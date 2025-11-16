import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import jax
jax.config.update("jax_platform_name","cpu")
jax.config.update("jax_enable_x64",True)
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


def main(ifol_num_epochs=10,solve_FE=False,solve_NiN=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "ifol_output_mechanical_2d"
    case_dir = os.path.join('.', working_directory_name)
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

    identity_control = IdentityControl('identity_control', control_settings={},num_vars=2*fe_mesh.GetNumberOfNodes())
    identity_control.Initialize()
    
    
    #### load a binary file:
    ## ATTENTION: fourier samples were produced for the phase contrast of 1 to 10
    ## but other samples can be modified here
    data_dir = os.path.join('.', working_directory_name+"_data")
    pc = 0.1
    
    with open(data_dir+'/Example_2d_info_res_21.pkl' , 'rb') as f:
        U_dict_21 = pickle.load(f)
    
    with open(data_dir+'/Example_2d_info_res_41.pkl' , 'rb') as f:
        U_dict_41 = pickle.load(f)

    with open(data_dir+'/Example_2d_info_res_81.pkl' , 'rb') as f:
        U_dict_81 = pickle.load(f)
    

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

    
    # load teh best model
    ifol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

    K_matrix = U_dict_81['K_matrix']
    load_and_plot = True
    if load_and_plot:
        topology = (U_dict_21["K_matrix"][eval_id,:], U_dict_41["K_matrix"][eval_id,:], U_dict_81["K_matrix"][eval_id,:])
        U_FE = (U_dict_21["U_FE"][eval_id,:][::2], U_dict_41["U_FE"][eval_id,:][::2], U_dict_81["U_FE"][eval_id,:][::2])
        U_NiN = (U_dict_21["U_NiN"][eval_id,:][::2], U_dict_41["U_NiN"][eval_id,:][::2], U_dict_81["U_NiN"][eval_id,:][::2])
        U_iFOL = (U_dict_21["U_iFOL"][eval_id,:][::2], U_dict_41["U_iFOL"][eval_id,:][::2], U_dict_81["U_iFOL"][eval_id,:][::2])
        error = (U_dict_21["U_error"][eval_id,:][::2], U_dict_41["U_error"][eval_id,:][::2], U_dict_81["U_error"][eval_id,:][::2])


        FE_stress_81 = get_stress(loss_function=mechanical_loss_2d, disp_field_vec= U_dict_81["U_NiN"][eval_id,:], K_matrix=np.array(K_matrix[eval_id,:]))
        iFOL_stress_81 = get_stress(loss_function=mechanical_loss_2d, disp_field_vec=U_dict_81["U_iFOL"][eval_id,:], K_matrix=np.array(K_matrix[eval_id,:]))
        stress_err_81 = abs(iFOL_stress_81.reshape(-1) - FE_stress_81.reshape(-1)).reshape(-1)

        plot_paper_triple(topology_field=topology, shape_tuple=(21,41,81), 
               fe_sol_field=U_NiN, ifol_sol_field=U_iFOL, 
               sol_field_err=error, file_name=case_dir+f'/tipple_plot_{eval_id}.png',
               fe_stress_field=FE_stress_81, ifol_stress_field=iFOL_stress_81, stress_field_err=stress_err_81)
    else:
        nan_indices, solved_indices, max_error_indices = [], [], []
        max_error,mean_error,std_error = [], [], []
        U_dict = {}

        for eval_id in [1,6,7,10,37,39,73,254,255]:

            ifol_uvw = np.array(ifol.Predict(K_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
            ifol_stress = get_stress(loss_function=mechanical_loss_2d, disp_field_vec=ifol_uvw, K_matrix=np.array(K_matrix[eval_id,:]))
            
            fe_mesh[f'iFOL_U_{eval_id}'] = ifol_uvw.reshape((fe_mesh.GetNumberOfNodes(), 2))
            fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(),1))
            fe_mesh[f"iFOL_stress_{eval_id}"] = ifol_stress.reshape((fe_mesh.GetNumberOfNodes(), 3))

            # solve FE here
            if solve_FE:
                fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                            "nonlinear_solver_settings":{"rel_tol":1e-6,"abs_tol":1e-6,
                                                            "maxiter":10,"load_incr":30},
                            "output_directory":case_dir}
                nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_2d,fe_setting)
                nonlin_fe_solver.Initialize()
                FE_UVW = np.array(nonlin_fe_solver.Solve(K_matrix[eval_id,:],np.zeros(2*fe_mesh.GetNumberOfNodes())))  
                plot_norm_iter(data=np.loadtxt(os.path.join(case_dir,"res_norm_jax.txt")),plot_name=case_dir+"/res_norm_iter_FE",type='1')

                fe_mesh[f'FE_U_{eval_id}'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))
                fe_stress = get_stress(loss_function=mechanical_loss_2d, disp_field_vec=FE_UVW, K_matrix=np.array(K_matrix[eval_id,:]))
                
                abs_err = abs(FE_UVW.reshape(-1,1) - ifol_uvw.reshape(-1,1))
                rmse_err = np.sqrt(np.mean(abs_err**2))
                abs_stress_err = abs(fe_stress.reshape(-1,1) - ifol_stress.reshape(-1,1))
                
                fe_mesh[f"abs_U_error_{eval_id}"] = abs_err.reshape((fe_mesh.GetNumberOfNodes(), 2))
                fe_mesh[f"FE_stress_{eval_id}"] = fe_stress.reshape((fe_mesh.GetNumberOfNodes(), 3))
                fe_mesh[f"abs_stress_error_{eval_id}"] = abs_stress_err.reshape((fe_mesh.GetNumberOfNodes(), 3))

                plot_iFOL_HFE(topology_field=K_matrix[eval_id,:], ifol_sol_field=ifol_uvw, hfe_sol_field=FE_UVW,
                    err_sol_field=abs_err, file_name=os.path.join(case_dir,'plots')+f"/ifol_fe-fe_error_{eval_id}",
                    fig_titles=['Elasticity Morph.','iFOL','FE','iFOL-FE Abs Diff.'])
                

            if solve_NiN:
                # hybrid solver
                fol_info("solve fe nin in one load step")
                nin_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                            "nonlinear_solver_settings":{"rel_tol":1e-6,"abs_tol":1e-6,
                                                            "maxiter":10,"load_incr":1},
                            "output_directory":case_dir}
                nin_nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("hybrid_nonlin_fe_solver",mechanical_loss_2d,nin_setting)
                nin_nonlin_fe_solver.Initialize()
                try:    
                    FE_NIN_UVW = np.array(nin_nonlin_fe_solver.Solve(K_matrix[eval_id,:],ifol_uvw.reshape(2*fe_mesh.GetNumberOfNodes())))  
                    # plot_norm_iter(data=np.loadtxt(os.path.join(case_dir,'res_norm_jax.txt')),
                    #                plot_name=os.path.join(case_dir,'plots')+f"\\res_norm_iter_NiN_{eval_id}")
                    solved_indices.append(eval_id)
                except Exception as e:
                    print(f"error occurd as: {type(e).__name__}")
                    FE_NIN_UVW = np.zeros(2*fe_mesh.GetNumberOfNodes())
                    nan_indices.append(eval_id)

                fe_mesh[f'NiN_U_{eval_id}'] = FE_NIN_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))
                fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(),1))
                
                U_dict[f'U_iFOL_res_{model_settings['N']}_{eval_id}'] = ifol_uvw
                U_dict[f"abs_error_res_{model_settings['N']}_{eval_id}"] = abs_err
                U_dict[f'K_matrix_res_{model_settings['N']}_{eval_id}'] = K_matrix[eval_id,:]
                U_dict[f'U_FE_res_{model_settings['N']}_{eval_id}'] = FE_UVW
                U_dict[f'U_NiN_res_{model_settings['N']}_{eval_id}'] = FE_NIN_UVW

                abs_err = abs(FE_UVW.reshape(-1,1) - ifol_uvw.reshape(-1,1))
                rmse_err = np.sqrt(np.mean(abs_err**2))
                std_err = np.std(abs_err**2)
                max_error.append(np.max(abs_err**2))
                mean_error.append(rmse_err)
                std_error.append(std_err)
                if float(np.max(abs_err)) > 0.4:
                    max_error_indices.append(eval_id)
                
                plot_iFOL_HFE(topology_field=K_matrix[eval_id,:], ifol_sol_field=ifol_uvw, hfe_sol_field=FE_NIN_UVW,
                     err_sol_field=abs_err, file_name=os.path.join(case_dir,'plots')+f"/ifol_fe-nin_error_{eval_id}",
                     fig_titles=['Elasticity Morph.','iFOL','FE-NIN','iFOL-FE Abs Diff.'])
                
        np.savetxt(os.path.join(case_dir,"nan_indices.txt"),np.array(nan_indices))
        np.savetxt(os.path.join(case_dir,"solved_indices.txt"),np.array(solved_indices))
        np.savetxt(os.path.join(case_dir,"max_error_l2.txt"),np.array(max_error))
        np.savetxt(os.path.join(case_dir,"rmse_error.txt"),np.array(mean_error))
        np.savetxt(os.path.join(case_dir,"std_error.txt"),np.array(std_error))
        np.savetxt(os.path.join(case_dir,"max_error_indices.txt"),np.array(max_error_indices))

        

    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)



if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 3000
    solve_FE = False
    solve_NiN = False
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
