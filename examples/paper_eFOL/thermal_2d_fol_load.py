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
from fol.deep_neural_networks.explicit_parametric_operator_learning import ExplicitParametricOperatorLearning
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle
from utilities import *

def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # cleaning & logging
    working_directory_name = 'nn_output_thermal_fol_2d_siren'
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


    fol_net = MLP(name='fol_net',input_size=fourier_control.GetNumberOfVariables(), 
                  output_size=thermal_loss_2d.GetNumberOfUnknowns(), hidden_layers=[300,300],
                  activation_settings={"type":"sin",
                                    "prediction_gain":5,
                                    "initialization_gain":1})

    # create fol optax-based optimizer
    chained_transform = optax.chain(optax.normalize_by_update_norm(), 
                                    optax.adam(1e-4))
    
    fol = ExplicitParametricOperatorLearning(name="dis_fol",control=fourier_control,
                                             loss_function=thermal_loss_2d,
                                             flax_neural_network=fol_net,
                                             optax_optimizer=chained_transform)

    fol.Initialize()
    train_start_id = 0
    train_end_id = 8000
    test_start_id = 8000
    test_end_id = 9000
    # fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
    #         #   test_set=(coeffs_matrix[test_start_id:test_end_id,:]),
    #                 convergence_settings={"num_epochs":fol_num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
    #                 plot_settings={"plot_save_rate":100},
    #                 train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
    #                 working_directory=case_dir)
    
    fol.RestoreState(restore_state_directory=case_dir+'/flax_train_state')
    fol_dict = {}
    FOL = np.array(fol.Predict(coeffs_matrix))
    fol_dict['exFOL_pred'] = FOL
    with open(f'thermal_fol_siren_pred_dict.pkl', 'wb') as f:
        pickle.dump(fol_dict,f)
    exit()
    for test in [9000,9121,9236,9347,9486,9563,9617,9785,9863,9963,9988,9999]:
        eval_id = test
        FOL_T = np.array(fol.Predict(coeffs_matrix[eval_id].reshape(-1,1).T)).reshape(-1)
        fe_mesh[f'T_FOL_{eval_id}'] = FOL_T.reshape((fe_mesh.GetNumberOfNodes(), 1))
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

            absolute_error = abs(FOL_T.reshape(-1,1)-FE_T.reshape(-1,1))
            relative_error = 100 * absolute_error/abs(FE_T.reshape(-1,1))
            fe_mesh[f'relative_error_{eval_id}'] = relative_error.reshape((fe_mesh.GetNumberOfNodes(), 1))
            fe_mesh[f'absolute_error_{eval_id}'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 1))

            plot_thermal_paper(vectors_list=[K_matrix[eval_id],FOL_T,FE_T], file_name=case_dir+f"/thermal_2d_sample_{eval_id}")

    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    fol_num_epochs = 5000
    solve_FE = True
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
            print("Usage: python thermal_fol.py fol_num_epochs=10 solve_FE=False clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(fol_num_epochs, solve_FE,clean_dir)