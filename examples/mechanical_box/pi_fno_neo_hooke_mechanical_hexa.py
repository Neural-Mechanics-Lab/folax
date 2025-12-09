import sys
import os

import numpy as np
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DHexa
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.controls.fourier_control import FourierControl
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.fourier_neural_operator_networks import FourierNeuralOperator3D
from fol.deep_neural_networks.fourier_parametric_operator_learning import PhysicsInformedFourierParametricOperatorLearning3D
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
import pickle
import optax
from flax.nnx import bridge
import pickle
from flax import nnx
import jax


def main(fol_num_epochs=10,solve_FE=False,clean_dir=False):
    # directory & save handling
    working_directory_name = "box_3D_hexa_nonlin"
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir,working_directory_name+".log"))

    # create mesh_io
    N = 20
    fe_mesh = create_3D_box_mesh_structured(Nx=N,Ny=N,Nz=N,Lx=1.,Ly=1.,Lz=1.)

    # creation of fe model and loss function
    bc_dict = {"Ux":{"left":0.0},
                "Uy":{"left":0.0,"right":-0.35},
                "Uz":{"left":0.0,"right":-0.35}}
    material_dict = {"young_modulus":1,"poisson_ratio":0.3}

    mechanical_loss_3d = NeoHookeMechanicalLoss3DHexa("mechanical_loss_3d",loss_settings={"dirichlet_bc_dict":bc_dict,
                                                                                   "material_dict":material_dict},
                                                                                   fe_mesh=fe_mesh)

    # fourier control
    fourier_control_settings = {"x_freqs":np.array([2,4,6]),"y_freqs":np.array([2,4,6]),"z_freqs":np.array([2,4,6]),
                                "beta":20,"min":1e-1,"max":1}
    fourier_control = FourierControl("fourier_control",fourier_control_settings,fe_mesh)

    fe_mesh.Initialize()
    mechanical_loss_3d.Initialize()
    fourier_control.Initialize()
    identity_control = IdentityControl("identity_control",control_settings={},num_vars= fe_mesh.GetNumberOfNodes())
    identity_control.Initialize()

    # create some random coefficients & K for training
    create_random_coefficients = True
    if create_random_coefficients:
        number_of_random_samples = 200
        coeffs_matrix,K_matrix = create_random_fourier_samples(fourier_control,number_of_random_samples)
        export_dict = {}
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

    # now save K matrix 
    export_Ks = False
    if export_Ks:
        for i in range(K_matrix.shape[0]):
            fe_mesh[f'K_{i}'] = np.array(K_matrix[i,:])
        fe_mesh.Finalize(export_dir=case_dir)

    eval_id = 69
    fe_mesh['K'] = np.array(K_matrix[eval_id,:])


    def merge_state(dst: nnx.State, src: nnx.State):
        for k, v in src.items():
            if isinstance(v, nnx.State):
                merge_state(dst[k], v)
            else:
                dst[k] = v

    fno_model = bridge.ToNNX(FourierNeuralOperator3D(modes1=8,
                                                    modes2=8,
                                                    modes3=8,
                                                    width=8,
                                                    depth=4,
                                                    channels_last_proj=128,
                                                    out_channels=3,
                                                    padding=4,
                                                    output_scale=0.001),rngs=nnx.Rngs(0)).lazy_init(K_matrix[0:1].reshape(1,N,N,N,1)) 

    # replace RNG key by a dummy to allow checkpoint restoration later
    graph_def, state = nnx.split(fno_model)
    rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))
    merge_state(state, rngs_key)
    fno_model = nnx.merge(graph_def, state)

    # get total number of fno params
    params = nnx.state(fno_model, nnx.Param)
    total_params  = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print(f"total number of fno network param:{total_params}")

    num_epochs = 1000
    learning_rate_scheduler = optax.linear_schedule(init_value=1e-5, end_value=1e-6, transition_steps=num_epochs)
    optimizer = optax.chain(optax.adam(learning_rate_scheduler))

    # create fol
    pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning3D(name="pi_fno_pr_learning",
                                                                            control=identity_control,
                                                                            loss_function=mechanical_loss_3d,
                                                                            flax_neural_network=fno_model,
                                                                            optax_optimizer=optimizer)

    pi_fno_pr_learning.Initialize()

    train_start_id = 0
    train_end_id = 180
    test_start_id = 180
    test_end_id = 200
    pi_fno_pr_learning.Train(train_set=(K_matrix[train_start_id:train_end_id,:],),
              batch_size=5,
              convergence_settings={"num_epochs":num_epochs,
                                    "relative_error":1e-100,
                                    "absolute_error":1e-100},
              train_checkpoint_settings={"least_loss_checkpointing":True,"frequency":100},
              working_directory=case_dir,
              plot_settings={"plot_list":["total_loss","phy1_loss","phy2_loss"],
                             "plot_frequency":1,"save_frequency":100,
                             "save_directory":".","multiphysics":True})

    eval_id = 0
    FNO_UVW = np.array(pi_fno_pr_learning.Predict(K_matrix[eval_id].reshape(-1,1).T)).reshape(-1)
    fe_mesh['U_FNO'] = FNO_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    # solve FE here
    if solve_FE:
        fe_setting = {"linear_solver_settings":{"solver":"PETSc-bcgsl"},
                      "nonlinear_solver_settings":{"rel_tol":1e-6,"abs_tol":1e-6,
                                                    "maxiter":5,"load_incr":40}}
        nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver",mechanical_loss_3d,fe_setting)
        nonlin_fe_solver.Initialize()
        FE_UVW = np.array(nonlin_fe_solver.Solve(K_matrix[eval_id],np.zeros(3*fe_mesh.GetNumberOfNodes())))  
        fe_mesh['U_FE'] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 3))

    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)

if __name__ == "__main__":
    # Initialize default values
    fol_num_epochs = 2000
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