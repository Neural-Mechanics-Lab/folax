import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import jax
jax.config.update("jax_platform_name","cpu")
jax.config.update("jax_enable_x64",True)
import jax.profiler
import numpy as np
from fol.loss_functions.mechanical_neohooke_AD import NeoHookeMechanicalLoss2DQuad
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
    working_directory_name = "ifol_output_FE"
    case_dir = os.path.join('.', working_directory_name)
    # create_clean_directory(working_directory_name)
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
            #    "Uy":{"bottom":model_settings["Uy_bottom"]}}
    
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}

    mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                        "num_gp":2,
                                                                                        "material_dict":material_dict},
                                                        fe_mesh=fe_mesh)

    mechanical_loss_2d.Initialize()

    identity_control = IdentityControl('identity_control', control_settings={},num_vars=fe_mesh.GetNumberOfNodes())
    identity_control.Initialize()
    

    tpms_settings = {"phi_x": 0., "phi_y": 0., "phi_z": 0., "max": 1., "min": 0.1, "section_axis_value": 0.5,
                     "constant": 0., "threshold": 0.5, "coefficients":(2.,2.,2.)}
    K_matrix = create_tpms_gyroid(fe_mesh=fe_mesh,tpms_settings=tpms_settings).reshape(-1,1).T
    # K_matrix = np.ones_like(K_matrix)

    U_dict = {}
    for eval_id in [0]:
        fe_setting = {"linear_solver_settings":{"solver":"JAX-direct"},
                    "nonlinear_solver_settings":{"rel_tol":1e-6,"abs_tol":1e-6,
                                                    "maxiter":20,"load_incr":31},
                            "output_directory":case_dir}
        nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_2d,fe_setting)
        nonlin_fe_solver.Initialize()
        jax.profiler.start_trace(case_dir+"/tmp/jax_profile_logs") # Start tracing, specify log directory
        FE_UVW = np.array(nonlin_fe_solver.Solve(K_matrix[eval_id,:],np.zeros(2*fe_mesh.GetNumberOfNodes()))) 
        jax.profiler.stop_trace() # Stop tracing and save logs
        exit()
        plot_norm_iter(data=np.loadtxt(os.path.join(case_dir,"res_norm_jax.txt")),plot_name=case_dir+"/res_norm_iter_FE",type='1') 
        fe_mesh[f'FE_U_{eval_id}'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))
        fe_mesh[f"K_{eval_id}"] = K_matrix[eval_id,:].reshape((fe_mesh.GetNumberOfNodes(), 1))

        U_dict[f'K_matrix_{eval_id}'] = K_matrix[eval_id,:]
        U_dict[f'U_FE_{eval_id}'] = FE_UVW

        # # save solution for base resolution as fields in a pkl file.  
        # with open(f"U_base_res_{model_settings['N']}_bc_{model_settings['Ux_right']}.pkl" , 'wb') as f:
        #     pickle.dump(U_dict,f)


    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)



if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 3000
    solve_FE = True
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
