import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
import optax
import numpy as np
from fol.loss_functions.mechanical import MechanicalLoss2DQuad
from fol.mesh_input_output.mesh import Mesh
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.solvers.fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from fol.deep_neural_networks.nns import HyperNetwork,MLP
from fol.tools.decoration_functions import *
from identity_control import IdentityControl
import pickle
import matplotlib.pyplot as plt 
from matplotlib.collections import PolyCollection


def plot_deformation_with_microstructure(fe_mesh, u, K_matrix, ax=None, scale=1.0, cmap=None):
    coords = np.asarray(fe_mesh.nodes_coordinates)[:, :2].copy()
    disp = np.asarray(u).reshape(-1,2)
    values = np.asarray(K_matrix)
    coords[:,1] = coords[:,1].max() - coords[:,1]

    coords_def = coords + scale * disp

    if "tri" in fe_mesh.elements_nodes:
        elements = fe_mesh.elements_nodes["tri"]
    elif "quad" in fe_mesh.elements_nodes:
        elements = fe_mesh.elements_nodes["quad"]
    else:
        raise ValueError("Only tri or quad elements supported for 2D plotting.")

    polys = []
    face_colors = []
    for elem in elements:
        elem_indices = np.array(elem, dtype=int)
        polys.append(coords_def[elem_indices])
        face_colors.append(values[elem_indices].mean())

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
        pc = PolyCollection(polys, array=np.array(face_colors), cmap=cmap, edgecolor="k")
        ax.add_collection(pc)
        fig.colorbar(pc, ax=ax, label="Young's Modulus")
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(f"Deformed microstructure (scale={scale})")
        plt.tight_layout()
        return fig, ax
    else:
        pc = PolyCollection(polys, array=np.array(face_colors), cmap=cmap, edgecolor="k")
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect("equal")
        return ax



def main(ifol_num_epochs=10,clean_dir=False, grid_size=None, external_K_matrix=None, fe_solver=True):

    if ifol_num_epochs<5000:
        fol_warning(f"ifol_num_epochs is set to {ifol_num_epochs}, recommended value for good results is 5000 !")

    # directory & save handling
    working_directory_name = 'meta_implicit_mechanical_2D'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # problem setup

    if grid_size is None:
        raise ValueError("Grid size (N) must be provided by the Streamlit app!")

    model_settings = {"L":1,"N":grid_size,  
                        "Ux_left":0.0,"Ux_right":0.05,
                        "Uy_left":0.0,"Uy_right":0.05}

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=model_settings["L"],N=model_settings["N"])

    # create fe-based loss function
    bc_dict = {"Ux":{"left":model_settings["Ux_left"],"right":model_settings["Ux_right"]},
            "Uy":{"left":model_settings["Uy_left"],"right":model_settings["Uy_right"]}}

    material_dict = {"young_modulus":1,"poisson_ratio":0.3}
    mechanical_loss_2d = MechanicalLoss2DQuad("mechanical_loss_2d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                "num_gp":2,
                                                                                "material_dict":material_dict},
                                                                                fe_mesh=fe_mesh)

    
    identity_control = IdentityControl('identity_control', control_settings={}, fe_mesh=fe_mesh)
    identity_control.Initialize()


    fe_mesh.Initialize()
    mechanical_loss_2d.Initialize()


    if external_K_matrix is not None:
        K_matrix = external_K_matrix
    elif os.path.exists("K_matrix.npy"):
        K_matrix = np.load("K_matrix.npy")
    else:
        raise ValueError("external_K_matrix must be provided.")
    
    # --- Save the undeformed microstructure as an image ---
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(K_matrix.reshape(model_settings["N"], model_settings["N"]),
                   origin='upper', aspect='equal', extent=(0, model_settings["L"], 0, model_settings["L"]))
    plt.colorbar(im, ax=ax, label="Young's Modulus (E)")
    ax.set_title("Microstructure")
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, "microstructure.png"), dpi=400, bbox_inches="tight")
    plt.close(fig)    

    export_Ks = False
    if export_Ks:
        for i in range(K_matrix.shape[0]):
            fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
        fe_mesh.Finalize(export_dir=case_dir)
        exit()



    # design synthesizer & modulator NN for hypernetwork
    characteristic_length = model_settings["N"]
    characteristic_length = 64
    synthesizer_nn = MLP(name="synthesizer_nn",
                        input_size=3,
                        output_size=2,
                        hidden_layers=[characteristic_length] * 6,
                        activation_settings={"type":"sin",
                                            "prediction_gain":60,
                                            "initialization_gain":1.0},
                        skip_connections_settings={"active":False,"frequency":1})

    latent_size = 2 * characteristic_length
    modulator_nn = MLP(name="modulator_nn",
                    input_size=latent_size,
                    use_bias=False) 

    hyper_network = HyperNetwork(name="hyper_nn",
                                modulator_nn=modulator_nn,synthesizer_nn=synthesizer_nn,
                                coupling_settings={"modulator_to_synthesizer_coupling_mode":"one_modulator_per_synthesizer_layer"})

    # create fol optax-based optimizer
    num_epochs = ifol_num_epochs
    # learning_rate_scheduler = optax.linear_schedule(init_value=1e-4, end_value=1e-7, transition_steps=num_epochs)
    # main_loop_transform = optax.chain(optax.normalize_by_update_norm(),optax.adam(learning_rate_scheduler))
    main_loop_transform = optax.chain(optax.adam(1e-5))
    latent_step_optimizer = optax.chain(optax.adam(1e-4))

    # create fol
    fol = MetaAlphaMetaImplicitParametricOperatorLearning(name="meta_implicit_fol",control=identity_control,
                                                            loss_function=mechanical_loss_2d,
                                                            flax_neural_network=hyper_network,
                                                            main_loop_optax_optimizer=main_loop_transform,
                                                            latent_step_optax_optimizer=latent_step_optimizer,
                                                            latent_step_size=1e-2,
                                                            num_latent_iterations=3)
    fol.Initialize()

    # train_start_id = 0
    # train_end_id = 1
    # test_start_id = 3*train_end_id
    # test_end_id = 3*train_end_id + 2

    # Only use the first sample for training
    train_start_id = 0
    train_end_id = 1   # just one sample

    # here we train for single sample at eval_id but one can easily pass the whole coeffs_matrix
    fol.Train(train_set=(K_matrix[train_start_id:train_end_id,:],),
            test_set=([],), # Skip test for on the fly 
            test_frequency=10,
            batch_size=1,
            convergence_settings={"num_epochs":num_epochs,
                                    "relative_error":1e-100,
                                    "absolute_error":1e-100},
            working_directory=case_dir)

    # load the best model
    fol.RestoreState(restore_state_directory=case_dir+"/flax_final_state")

    for test in range(train_start_id,train_end_id):
        # eval_id = test
        eval_id = 0
        FOL_UV = np.array(fol.Predict(K_matrix[eval_id,:].reshape(-1,1).T)).reshape(-1)
        fe_mesh['U_FOL'] = FOL_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

        if fe_solver:

            # solve FE here
            fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl","tol":1e-6,"atol":1e-6,
                                                        "maxiter":1000,"pre-conditioner":"ilu"},
                            "nonlinear_solver_settings":{"rel_tol":1e-5,"abs_tol":1e-5,
                                                        "maxiter":10,"load_incr":5}}
            linear_fe_solver = FiniteElementLinearResidualBasedSolver("linear_fe_solver",mechanical_loss_2d,fe_setting)
            linear_fe_solver.Initialize()
            FE_UV = np.array(linear_fe_solver.Solve(K_matrix[eval_id],np.zeros(2*fe_mesh.GetNumberOfNodes())))  
            fe_mesh['U_FE'] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

            absolute_error = abs(FOL_UV.reshape(-1,1)- FE_UV.reshape(-1,1))
            fe_mesh['abs_error'] = absolute_error.reshape((fe_mesh.GetNumberOfNodes(), 2))


            plot_mesh_vec_data(1,[FOL_UV[0::2],FOL_UV[1::2],absolute_error[0::2],absolute_error[1::2]],
                            ["U","V","abs_error_U","abs_error_V"],
                            fig_title="implicit FOL solution and error",
                            # cmap="coolwarm",
                            file_name=os.path.join(case_dir,f"FOL-UV-dist_test_{eval_id}.png"))
            # plot_mesh_vec_data(
            #     1,
            #     [
            #         K_matrix[eval_id,:],       # Top-left: K
            #         FOL_UV[0::2],              # Top-right: U (FOL)
            #         absolute_error[0::2],      # Bottom-left: |U FEM - FOL|
            #         FE_UV[0::2]                # Bottom-right: U (FEM)
            #     ],
            #     ["K", "U (FOL)", "abs error U", "U (FEM)"],
            #     fig_title="iFOL vs FEM U, and Error",
            #     cmap="coolwarm",
            #     file_name=os.path.join(case_dir, f"K-FOL-FEMU-ERR-dist_test_{eval_id}.png")
            #     )
            # plot_mesh_vec_data(1,[K_matrix[eval_id,:],FE_UV[0::2],FE_UV[1::2]],
            #                 ["K","U","V"],
            #                 fig_title="conductivity and FEM solution",
            #                 cmap="coolwarm",
            #                 file_name=os.path.join(case_dir,f"FEM-KUV-dist_test_{eval_id}.png"))
            
            K_matrix_nodes = np.ravel(K_matrix[eval_id, :])
            # plot_deformation_with_microstructure(fe_mesh, FE_UV, K_matrix_nodes, scale=5.0, file_name="FE_deformed_microstructure.png")
            # plot_deformation_with_microstructure(fe_mesh, FE_UV, K_matrix_nodes, scale=5.0, file_name=os.path.join(case_dir,f"FE_deformed_microstructure_{eval_id}.png"))
            # plot_deformation_with_microstructure(fe_mesh, FOL_UV, K_matrix_nodes, scale=5.0, file_name="FOL_deformed_microstructure.png")
            # plot_deformation_with_microstructure(fe_mesh, FOL_UV, K_matrix_nodes, scale=5.0, file_name=os.path.join(case_dir,f"FOL_deformed_microstructure_{eval_id}.png"))

            #side-by-side comparison
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # FE deformation
            plot_deformation_with_microstructure(fe_mesh, FE_UV, K_matrix_nodes, ax=axes[0], scale=5.0)
            axes[0].set_title("FE Deformation")

            # FOL deformation
            plot_deformation_with_microstructure(fe_mesh, FOL_UV, K_matrix_nodes, ax=axes[1], scale=5.0)
            axes[1].set_title("FOL Deformation")

            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, f"FE_FOL_deformation_{eval_id}.png"), dpi=600)
            plt.close(fig)
            
            # vectors_list = [K_matrix[eval_id],FE_UV[::2],FOL_UV[::2]]
            # plot_mesh_res(vectors_list, file_name=os.path.join(case_dir,'plot_U.png'),dir="U")
            # plot_mesh_grad_res_mechanics(vectors_list, file_name=os.path.join(case_dir,'plot_stress_U.png'), loss_settings=material_dict)
            
            # vectors_list = [K_matrix[eval_id],FE_UV[1::2],FOL_UV[1::2]]
            # plot_mesh_res(vectors_list, file_name=os.path.join(case_dir,'plot_V.png'),dir="V")
            # plot_mesh_grad_res_mechanics(vectors_list, file_name=os.path.join(case_dir,'plot_stress_V.png'), loss_settings=material_dict)
        else:
            print("Skipping FE solver")
            plot_mesh_vec_data(1,
                           [FOL_UV[0::2], FOL_UV[1::2]],
                           ["U","V"],
                           fig_title="FEM  and conductivity solution",
                           #cmap="coolwarm",
                           file_name=os.path.join(case_dir,f"FOL-UV-only_test_{eval_id}.png"))
            
            K_matrix_nodes = np.ravel(K_matrix[eval_id, :])

            fig, ax = plt.subplots(figsize=(6, 6))

            # FOL deformation only
            plot_deformation_with_microstructure(fe_mesh, FOL_UV, K_matrix_nodes, ax=ax, scale=5.0)
            ax.set_title("FOL Deformation")

            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, f"FOL_deformation_{eval_id}.png"), dpi=600)
            plt.close(fig)
            # plot_deformation_with_microstructure(fe_mesh, FOL_UV, K_matrix_nodes, scale=5.0, file_name="FOL_deformed_microstructure.png")
            


    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)   


if __name__ == "__main__":
    # Initialize default values
    ifol_num_epochs = 2
    clean_dir = False
    grid_size = None  # default
    fe_solver = True

    # Parse the command-line arguments
    args = sys.argv[1:]

    # Process the arguments if provided
    # for arg in args:
    #     if arg.startswith("ifol_num_epochs="):
    #         try:
    #             ifol_num_epochs = int(arg.split("=")[1])
    #         except ValueError:
    #             print("ifol_num_epochs should be an integer.")
    #             sys.exit(1)
    #     elif arg.startswith("clean_dir="):
    #         value = arg.split("=")[1]
    #         if value.lower() in ['true', 'false']:
    #             clean_dir = value.lower() == 'true'
    #         else:
    #             print("clean_dir should be True or False.")
    #             sys.exit(1)
    #     else:
    #         print("Usage: python script.py ifol_num_epochs=10 clean_dir=False N=10")
    #         sys.exit(1)
    for arg in args:
        if arg.startswith("ifol_num_epochs="):
            ifol_num_epochs = int(arg.split("=")[1])
        elif arg.startswith("clean_dir="):
            value = arg.split("=")[1]
            clean_dir = value.lower() == 'true'
        elif arg.startswith("N="):
            grid_size = int(arg.split("=")[1])
        elif arg.startswith("fe_solver="):
            fe_solver = arg.split("=")[1].lower() == 'true'

    # Call the main function with the parsed values
    main(ifol_num_epochs, clean_dir, grid_size, fe_solver=fe_solver)


