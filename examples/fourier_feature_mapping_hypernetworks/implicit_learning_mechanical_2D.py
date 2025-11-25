import sys
import os
import optax
import numpy as np
from fol.loss_functions.mechanical import MechanicalLoss2DQuad
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.tools.decoration_functions import *
import pickle

def main(ifol_num_epochs=10,clean_dir=False):

    if ifol_num_epochs<5000:
        fol_warning(f"ifol_num_epochs is set to {ifol_num_epochs}, recommended value for good results is 5000 !")

    # ---------------------------
    # I/O and logging
    # ---------------------------
    working_directory_name = 'implicit_learning_mechanical_2D'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # ---------------------------
    # Problem setup (2D square, small displacement BCs)
    # ---------------------------
    model_settings = {"L":1,"N":11,
                        "Ux_left":0.0,"Ux_right":0.05,
                        "Uy_left":0.0,"Uy_right":0.05}

    # Build a structured 2D square mesh of size L with N nodes/edge
    fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

    # ---------------------------
    # FE-based loss: linear elasticity with Dirichlet BCs on Ux, Uy
    # ---------------------------
    bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
            "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}

    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    mechanical_loss_2d = MechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "num_gp":2,
                                                                                "material_dict":material_dict},
                                                                                fe_mesh=fe_mesh)

    # ---------------------------
    # Parametric control field via Fourier basis (design variables → heterogeneity K)
    # ---------------------------
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([0]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)


     # Initialize mesh, loss, and control
    fe_mesh.Initialize()
    mechanical_loss_2d.Initialize()
    fourier_control.Initialize()

    # ---------------------------
    # Load precomputed Fourier coefficients and produce heterogeneous fields
    # ---------------------------
    with open(f'fourier_control_dict.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
    
    coeffs_matrix = loaded_dict["coeffs_matrix"]
    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

    # ---------------------------
    # Hypernetwork setup
    #  - synthesizer_nn: maps (x,y,z) to displacements (U,V)
    #  - modulator_nn: produces latent(s) that modulate synthesizer layers
    #
    # KEY POINT: Enable Fourier features on the synthesizer + ReLU activation
    #            to capture high-frequency variations in the solution field.
    #
    # Why it helps: Random Fourier features (Gaussian) map inputs to a
    # high-dimensional oscillatory basis. After this mapping, a ReLU MLP can
    # linearly combine these features to represent sharp/oscillatory patterns
    # much more easily than with raw coordinates alone.
    # ---------------------------
    characteristic_length = 64
    synthesizer_nn = MLP(name="synthesizer_nn",
                         input_size=3,
                         output_size=2,
                         hidden_layers=[characteristic_length] * 6,
                         activation_settings={"type":"relu"},
                         skip_connections_settings={"active":False,"frequency":1},
                         fourier_feature_settings={"active":True, # turn on Fourier features here
                                                   "size":characteristic_length, # number of random features
                                                   "frequency_scale":10.0,  # ↑ scale → higher effective frequencies
                                                   "learn_frequency":False}) # fixed (Gaussian) random features

    latent_size = 2 * characteristic_length
    modulator_nn = MLP(name="modulator_nn",
                       input_size=latent_size,
                       use_bias=False,
                       fourier_feature_settings={"active":False,        # keep modulator simple; no FF
                                                   "size":latent_size,
                                                   "frequency_scale":1.0,
                                                   "learn_frequency":False}) 

    # Couple modulator ↔ synthesizer (e.g., FiLM-like or per-layer scaling)
    hyper_network = HyperNetwork(name="hyper_nn",
                                 modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                                 coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

    # ---------------------------
    # Optimizer and Meta-Implicit OL wrapper
    # ---------------------------
    num_epochs = ifol_num_epochs
    main_loop_transform = optax.chain(optax.adam(1e-5))

    # create fol
    fol = MetaImplicitParametricOperatorLearning(name="meta_implicit_ol",control=fourier_control,
                                                loss_function=mechanical_loss_2d,
                                                flax_neural_network=hyper_network,
                                                main_loop_optax_optimizer=main_loop_transform,
                                                latent_step_size=1e-2,
                                                num_latent_iterations=3)
    fol.Initialize()

    # ---------------------------
    # Training: a small subset for demo
    # ---------------------------
    train_start_id = 0
    train_end_id = 20
    # Train on a batch of coefficient vectors (could pass the whole matrix too)
    fol.Train(train_set=(coeffs_matrix[train_start_id:train_end_id,:],),
                batch_size=1,
                convergence_settings={"num_epochs":num_epochs,
                                        "relative_error":1e-100,
                                        "absolute_error":1e-100},
                train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
                working_directory=case_dir)

    # Load best checkpoint (least-loss)
    fol.RestoreState(restore_state_directory=case_dir+"/flax_train_state")

    # ---------------------------
    # Evaluation: compare implicit model vs. classical FEM
    # ---------------------------
    for test in range(train_start_id,train_end_id):
        eval_id = test
        # Predict displacement with the learned implicit operator (FOL)
        FOL_UV = np.array(fol.Predict(coeffs_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
        fe_mesh['U_FOL'] = FOL_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

         # Solve reference FEM for the same K field
        fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl","tol":1e-6,"atol":1e-6,
                                                    "maxiter":1000,"pre-conditioner":"ilu"}}
        linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_2d,fe_setting)
        linear_fe_solver.Initialize()
        FE_UV = np.array(linear_fe_solver.Solve(K_matrix[eval_id],np.zeros(2*fe_mesh.GetNumberOfNodes())))  
        fe_mesh['U_FE'] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

        absolute_error = abs(FOL_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
        fe_mesh['abs_error'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 2))

        # Plots: model prediction + error

        plot_mesh_vec_data(1,[FOL_UV[0::2],FOL_UV[1::2],absolute_error[0::2],absolute_error[1::2]],
                        ["U","V","abs_error_U","abs_error_V"],
                        fig_title="implicit FOL solution and error",
                        file_name=os.path.join(case_dir,f"FOL-UV-dist_test_{eval_id}.png"))
        plot_mesh_vec_data(1,[K_matrix[eval_id,:],FE_UV[0::2],FE_UV[1::2]],
                        ["K","U","V"],
                        fig_title="conductivity and FEM solution",
                        file_name=os.path.join(case_dir,f"FEM-KUV-dist_test_{eval_id}.png"))

    # Export any mesh-attached fields
    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)   


if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 5000
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
        elif arg.startswith("clean_dir="):
            value = arg.split("=")[1]
            if value.lower() in ['true', 'false']:
                clean_dir = value.lower() == 'true'
            else:
                print("clean_dir should be True or False.")
                sys.exit(1)
        else:
            print("Usage: python script.py ifol_num_epochs=10 clean_dir=False")
            sys.exit(1)

    # Call the main function with the parsed values
    main(ifol_num_epochs, clean_dir)