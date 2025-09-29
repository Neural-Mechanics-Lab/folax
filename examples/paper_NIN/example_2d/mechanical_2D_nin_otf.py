import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import jax
# jax.config.update("jax_platform_name","cpu")
# jax.config.update("jax_enable_x64",True)
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
### to retrain the network to reach to a specific solution by performing an OTF on the sample.


def main(ifol_num_epochs=10,solve_FE=False,solve_HFE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "ifol_output_mechanical_2d_otf"
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

    identity_control = IdentityControl('identity_control', control_settings={}, fe_mesh=fe_mesh)
    identity_control.Initialize()
    

    #### load a txt file:
    data_dir = os.path.join('.', "ifol_output_mechanical_2d_data")
    pc = 0.1
    fourier_file = os.path.join(data_dir,f"K_matrix_fourier_res_{model_settings['N']}.txt")
    voronoi_file = os.path.join(data_dir,f"K_matrix_voronoi_res_{model_settings['N']}.txt")
    voronoi_multi_file = os.path.join(data_dir,f"K_matrix_voronoi_multi_res_{model_settings['N']}.txt")
    tpms_file = os.path.join(data_dir,f"K_matrix_tpms_res_{model_settings['N']}_PC.txt")
    
    # tpms_settings = {"phi_x": 0., "phi_y": 0., "phi_z": 0., "max": 1., "min": 0.1, "section_axis_value": 1.0,
    #                  "constant": 0., "threshold": 0.5, "coefficients":(2.,2.,2.)}
    # K_matrix = create_tpms_gyroid(fe_mesh=fe_mesh,tpms_settings=tpms_settings).reshape(-1,1).T

    # tpms_file = os.path.join(case_dir,f"ifol_tpms_test_samples_K_matrix_res_{model_settings['N']}.txt")
    

    K_matrix_fourier = np.loadtxt(fourier_file)
    print("minimum value of K matrix : ",np.min(K_matrix_fourier))
    K_matrix_voronoi_multi = np.loadtxt(voronoi_multi_file)

    K_matrix_voronoi = np.loadtxt(voronoi_file)
    for i in range(K_matrix_voronoi.shape[0]):
        K_matrix_voronoi[i,:] = np.where(K_matrix_voronoi[i,:] < 1., pc, 1.)
    
    K_matrix_tpms = np.loadtxt(tpms_file)
    for i in range(K_matrix_tpms.shape[0]):
        K_matrix_tpms[i,:] = np.where(K_matrix_tpms[i,:] < 1., pc, 1.)

    K_matrix_1 = np.vstack((K_matrix_fourier,K_matrix_voronoi))
    K_matrix = np.vstack((K_matrix_1, K_matrix_tpms))


    # now we need to create, initialize and train fol
    ifol_settings_dict = {
        "characteristic_length": 64,
        "synthesizer_depth": 4,
        "activation_settings":{"type":"sin",
                                "prediction_gain":30,
                                "initialization_gain":1.0},
        "skip_connections_settings": {"active":False,"frequency":1},
        "latent_size":  1*64,
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

    otf_id = 272
    train_set_otf = K_matrix[otf_id,:].reshape(-1,1).T     # for On The Fly training

    # load teh best model
    # ifol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

    ifol.Train(train_set=(train_set_otf,),
                test_set=(train_set_otf,),
                test_frequency=100,
                batch_size=1,
                convergence_settings={"num_epochs":ifol_num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                plot_settings={"plot_save_rate":100},
                train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":10},
                working_directory=case_dir)


    # load teh best model
    ifol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

    ifol_solution_dict = {}
    U_dict_base = {}
    with open(os.path.join(data_dir,f'U_base_res_{model_settings["N"]}_PC_{pc}_revised.pkl'), 'rb') as f:
        U_dict_base = pickle.load(f)
    indices = []
    for i in range(K_matrix.shape[0]):
        if U_dict_base.get(f"U_FE_{model_settings['N']}_{i}") is not None:
            indices.append(i)
    


    for eval_id in [otf_id]:
        ifol_uvw = np.array(ifol.Predict(K_matrix[eval_id].reshape(-1,1).T)).reshape(-1)
        ifol_stress = get_stress(loss_function=mechanical_loss_2d, disp_field_vec=ifol_uvw, K_matrix=np.array(K_matrix[eval_id,:]))
        
        fe_mesh[f'iFOL_U_{eval_id}'] = ifol_uvw.reshape((fe_mesh.GetNumberOfNodes(), 2))
        fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(),1))
        fe_mesh[f"iFOL_stress_{eval_id}"] = ifol_stress.reshape((fe_mesh.GetNumberOfNodes(), 3))
        
        ifol_solution_dict[f'sample_{eval_id}'] = {}
        ifol_solution_dict[f'sample_{eval_id}'] = {
            "bc_dict": bc_dict,
            "eval_id": eval_id,
            "model_settings": model_settings,
            "solution_field": ifol_uvw,
            "First_Piola_Kirchhoff_field": ifol_stress,
            "total_K_matrix": K_matrix
        }
        with open(os.path.join(case_dir,'ifol_solution.pkl'), 'wb') as f:
            pickle.dump(ifol_solution_dict,f)

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
            fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(),1))
            fe_stress = get_stress(loss_function=mechanical_loss_2d, disp_field_vec=FE_UVW, K_matrix=np.array(K_matrix[eval_id,:]))
            
            abs_err = abs(FE_UVW.reshape(-1,1) - ifol_uvw.reshape(-1,1))
            abs_stress_err = abs(fe_stress.reshape(-1,1) - ifol_stress.reshape(-1,1))
            
            fe_mesh[f"abs_U_error_{eval_id}"] = abs_err.reshape((fe_mesh.GetNumberOfNodes(), 2))
            fe_mesh[f"FE_stress_{eval_id}"] = fe_stress.reshape((fe_mesh.GetNumberOfNodes(), 3))
            fe_mesh[f"abs_stress_error_{eval_id}"] = abs_stress_err.reshape((fe_mesh.GetNumberOfNodes(), 3))


            plot_iFOL_HFE(topology_field=K_matrix[eval_id,:], ifol_sol_field=ifol_uvw, hfe_sol_field=FE_UVW,
                 err_sol_field=abs_err, file_name=os.path.join(case_dir,'plots')+f"\\ifol_fe_error_{eval_id}",
                 fig_titles=['Elasticity Morph.','iFOL','FE'])
            
        # solve FE-NIN here
        if solve_HFE:
            hfe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                        "nonlinear_solver_settings":{"rel_tol":1e-6,"abs_tol":1e-6,
                                                        "maxiter":10,"load_incr":1},
                        "output_directory":case_dir}
            nonlin_hfe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_hfe_solver",mechanical_loss_2d,hfe_setting)
            nonlin_hfe_solver.Initialize()
            try:
                HFE_UVW = np.array(nonlin_hfe_solver.Solve(K_matrix[eval_id,:],np.zeros(2*fe_mesh.GetNumberOfNodes())))  
            except:
                ValueError("res_norm contains nan values!")
                HFE_UVW = np.zeros(2*fe_mesh.GetNumberOfNodes())

            plot_norm_iter(data=np.loadtxt(os.path.join(case_dir,"res_norm_jax.txt")),plot_name=case_dir+"/res_norm_iter_FE",type='1')

            fe_mesh[f'HFE_U_{eval_id}'] = HFE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))
            abs_err = abs(HFE_UVW.reshape(-1,1) - ifol_uvw.reshape(-1,1))

            plot_iFOL_HFE(topology_field=K_matrix[eval_id,:], ifol_sol_field=ifol_uvw, hfe_sol_field=HFE_UVW,
                 err_sol_field=abs_err, file_name=os.path.join(case_dir,'plots')+f"\\ifol_nin_error_{eval_id}",
                 fig_titles=['Elasticity Morph.','iFOL','FE-NIN', 'iFOL FE-NIN Abs Difference'])
            

    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)



if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 500
    solve_FE = False
    solve_HFE = True
    clean_dir = False

    # Parse the command-line arguments
    args = sys.argv[1:]

    # Process the arguments if provided
    for arg in args:
        if arg.startswith("ifol_num_epochs="):
            try:
                ifol_num_epochs = int(arg.split("=")[1])
            except ValueError:
                print("ifol_num_epochs should be an integer.")
                sys.exit(1)
        elif arg.startswith("solve_FE="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                solve_FE = value.lower() == 'true'
            else:
                print("solve_FE should be True or False.")
                sys.exit(1)
        elif arg.startswith("solve_HFE="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                solve_HFE = value.lower() == 'true'
            else:
                print("solve_HFE should be True or False.")
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
    main(ifol_num_epochs,solve_FE,solve_HFE,clean_dir)
