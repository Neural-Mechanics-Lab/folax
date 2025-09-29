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

### Script's goal:
####### SHOULD BE DELETED!


def main(ifol_num_epochs=10,solve_FE=False,solve_FE_hybrid=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "ifol_output_mechanical_2d"
    case_dir = os.path.join('.', working_directory_name)
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
    data_dir = os.path.join('.', working_directory_name+"_data")
    pc = 0.1
    fourier_file = os.path.join(data_dir,f"K_matrix_fourier_res_{model_settings['N']}.txt")
    voronoi_file = os.path.join(data_dir,f"K_matrix_voronoi_res_{model_settings['N']}.txt")
    voronoi_multi_file = os.path.join(data_dir,f"K_matrix_voronoi_multi_res_{model_settings['N']}.txt")
    tpms_file = os.path.join(data_dir,f"K_matrix_tpms_res_{model_settings['N']}.txt")
    
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

    U_dict_base = {}
    with open(os.path.join(data_dir,f'U_base_res_{model_settings["N"]}_PC_{pc}.pkl'), 'rb') as f:
        U_dict_base = pickle.load(f)
    
    # indices = []
    # for i in range(K_matrix.shape[0]):
    #     if U_dict_base.get(f"U_FE_{model_settings['N']}_{i}") is not None:
    #         indices.append(i)
    indices = np.loadtxt(os.path.join(".",'ifol_output_mechanical_2d/max_error_indices.txt'))
    indices = indices.astype(np.int32)

    fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                "nonlinear_solver_settings":{"rel_tol":1e-6,"abs_tol":1e-6,
                                                "maxiter":10,"load_incr":40},
                "output_directory":case_dir}
    nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_2d,fe_setting)
    nonlin_fe_solver.Initialize()

    for eval_id in indices:
        FE_UVW = np.array(nonlin_fe_solver.Solve(K_matrix[eval_id,:],np.zeros(2*fe_mesh.GetNumberOfNodes())))  
        U_dict_base[f"U_FE_{model_settings['N']}_{eval_id}"] = FE_UVW
        fe_mesh[f'FE_U_{eval_id}'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))
        fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(),1))

    with open(os.path.join(data_dir,f'U_base_res_{model_settings["N"]}_PC_{pc}_revised.pkl'), 'wb') as f:
        pickle.dump(U_dict_base,f)

    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)



if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 3000
    solve_FE = False
    solve_FE_hybrid = True
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
