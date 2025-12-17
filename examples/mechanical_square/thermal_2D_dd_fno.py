import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..',)))
import optax
from flax import nnx
import jax
import numpy as np
from fol.loss_functions.regression_loss import RegressionLoss
from fol.controls.identity_control import IdentityControl
from fol.controls.fourier_control import FourierControl
from fol.loss_functions.thermal import ThermalLoss2DQuad
from fol.deep_neural_networks.fourier_parametric_operator_learning import DataDrivenFourierParametricOperatorLearning
from fol.deep_neural_networks.fourier_neural_operator_networks import FourierNeuralOperator2D
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
import pickle, time
from flax.nnx import bridge


def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = 'nn_output_thermal_2D_data_driven_fno'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup
    model_settings = {"L":1,"N":42,
                    "T_left":1.0,"T_right":0.1}

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])
    
    # create fe-based loss function
    bc_dict = {"T":{"left":model_settings["T_left"],"right":model_settings['T_right']}}

    thermal_loss_2d = ThermalLoss2DQuad("thermal_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict, 
                                                                        "num_gp":2,
                                                                        "beta":2,"c":4},
                                                                        fe_mesh=fe_mesh)
    # fourier control
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

    fe_mesh.Initialize()
    thermal_loss_2d.Initialize()
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

    # create identity control
    identity_control = IdentityControl("ident_control",control_settings={},num_vars=K_matrix.shape[1])

    # create regression loss
    reg_loss = RegressionLoss("reg_loss",loss_settings={"nodal_unknows":['T']},fe_mesh=fe_mesh)

    # initialize all 
    reg_loss.Initialize()
    identity_control.Initialize()

    dof = len(thermal_loss_2d.GetDOFs())
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
                                                    out_channels=dof,
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

    num_epochs = 2000
    learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-5, transition_steps=num_epochs)
    optimizer = optax.chain(optax.adam(1e-3))

    # create fol
    pi_fno_pr_learning = DataDrivenFourierParametricOperatorLearning(name="pi_fno_pr_learning",
                                                                            control=identity_control,
                                                                            loss_function=reg_loss,
                                                                            flax_neural_network=fno_model,
                                                                            optax_optimizer=optimizer)

    pi_fno_pr_learning.Initialize()
    
    # create data set 
    gt_dataset_dict = {}
    create_dataset = False
    if create_dataset:
        number_of_samples = 200
        gt_dataset = np.zeros((number_of_samples,thermal_loss_2d.GetTotalNumberOfDOFs()))
        assert gt_dataset.shape[0] <= K_matrix.shape[0], "Number of data should be less than the number of samples!"
        fe_setting = {"linear_solver_settings":{"solver":"JAX-bicgstab","tol":1e-6,"atol":1e-6,
                                                    "maxiter":1000,"pre-conditioner":"ilu"}}
        lin_fe_solver = FiniteElementLinearResidualBasedSolver("lin_fe_solver",thermal_loss_2d,fe_setting)
        lin_fe_solver.Initialize()
        for i in range(number_of_samples):
            FE_T = np.array(lin_fe_solver.Solve(K_matrix[i],np.zeros(dof*fe_mesh.GetNumberOfNodes())))  
            gt_dataset[i,:] = FE_T.flatten()
        
        gt_dataset_dict['K_matrix'] = K_matrix[:number_of_samples,:]
        gt_dataset_dict['coeffs_matrix'] = coeffs_matrix[:number_of_samples,:]
        gt_dataset_dict['FE_T'] = gt_dataset
        with open("ground_truth_dataset.pkl", 'wb') as f:
            pickle.dump(gt_dataset_dict,f)

    else:
        with open('ground_truth_dataset.pkl', 'rb') as f:
            gt_dataset_dict = pickle.load(f)
    
    
    # split test train dataset
    otf_id = 0
    train_start_id = 0
    train_end_id = 10
    test_start_id = 180
    test_end_id = 182

    # Parametric learning or On The Fly learning
    parametric_learning = True
    if parametric_learning:
        train_set = K_matrix[train_start_id:train_end_id,:]
        test_set = K_matrix[test_start_id:test_end_id,:]
        eval_cases = range(test_start_id,test_end_id)
        batch_size = 5
        gt_train_set = gt_dataset_dict['FE_T'][train_start_id:train_end_id,:]
        gt_test_set = gt_dataset_dict['FE_T'][test_start_id:test_end_id,:]
    else:
        train_set = K_matrix[otf_id,:].reshape(-1,1).T
        test_set = K_matrix[otf_id,:].reshape(-1,1).T
        eval_cases = [otf_id]
        batch_size = 1
        gt_train_set = gt_dataset_dict['FE_T'][otf_id,:].reshape(-1,1).T
        gt_test_set = gt_dataset_dict['FE_T'][otf_id,:].reshape(-1,1).T

    #here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
    pi_fno_pr_learning.Train(train_set=(train_set,gt_train_set),
                            test_set=(test_set,gt_test_set),
                            test_frequency=100,
                            batch_size=batch_size,
                            convergence_settings={"num_epochs":num_epochs,"relative_error":1e-100,"absolute_error":1e-100},
                            plot_settings={"plot_save_rate":100},
                            train_checkpoint_settings={"least_loss_checkpointing":False,"frequency":100},
                            working_directory=case_dir)

    # load teh best model
    pi_fno_pr_learning.RestoreState(restore_state_directory=case_dir+"/flax_final_state")


    for eval_id in eval_cases:
        FNO_T = np.array(pi_fno_pr_learning.Predict(K_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
        fe_mesh[f'U_FNO_{eval_id}'] = FNO_T.reshape((fe_mesh.GetNumberOfNodes(), dof))

        FE_T = gt_dataset_dict['FE_T'][eval_id,:]
        fe_mesh[f'U_FE_{eval_id}'] = FE_T.reshape((fe_mesh.GetNumberOfNodes(), dof))

        absolute_error = abs(FNO_T.reshape(-1,1)- FE_T.reshape(-1,1))
        fe_mesh[f'abs_error_{eval_id}'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), dof))
        
        plot_mesh_vec_data(model_settings["L"], [K_matrix[eval_id,:],FNO_T,FE_T,absolute_error], 
                        subplot_titles= ['Conductivity', 'FNO_T', 'FE_T', "absolute_error"], fig_title=None, cmap='viridis',
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