import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import shutil

import optax
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import (
    FiniteElementNonLinearResidualBasedSolver,
)
from fol.controls.fourier_control import FourierControl
from fol.controls.identity_control import IdentityControl
from fol.deep_neural_networks.fourier_parametric_operator_learning import (
    PhysicsInformedFourierParametricOperatorLearning,
)
from fol.deep_neural_networks.fourier_neural_operator_networks import FourierNeuralOperator2D
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
import pickle, time
from flax.nnx import bridge
from newton_residual_tracker import custom_newton_solve   # Newton tracker


def main(
    fol_num_epochs=10,
    solve_FE=False,
    clean_dir=False,
    restore_param_init=True,
    parametric_learning=False,
    do_zssr=False,
    zssr_N=82,
    zssr_only=False,
):
    # ============================================================
    # 0) directory & logging
    # ============================================================
    working_directory_name = '1_Param_to_OTF_Transfer_ZSSR_reference_circular'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    os.makedirs(case_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(case_dir, working_directory_name + ".log"))

    # ============================================================
    # 1) problem setup (base resolution)
    # ============================================================
    model_settings = {
        "L": 1.0,
        "N": 42,
        "Ux_left": 0.0,
        "Ux_right": 0.1,
        "Uy_left": 0.0,
        "Uy_right": 0.1,
    }

    # mesh (base resolution)
    fe_mesh = create_2D_square_mesh(L=model_settings["L"], N=model_settings["N"])

    # FE loss (Neo-Hooke)
    bc_dict = {
        "Ux": {
            "left": model_settings["Ux_left"],
            "right": model_settings["Ux_right"],
        },
        "Uy": {
            "left": model_settings["Uy_left"],
            "right": model_settings["Uy_right"],
        },
    }

    material_dict = {"young_modulus": 1.0, "poisson_ratio": 0.3}
    mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad(
        "mechanical_loss_2d",
        loss_settings={
            "dirichlet_bc_dict": bc_dict,
            "num_gp": 2,
            "material_dict": material_dict,
        },
        fe_mesh=fe_mesh,
    )

    # Fourier control – ONLY used to regenerate the original K(x,y) fields
    fourier_control_settings = {
        "x_freqs": np.array([2, 4, 6]),
        "y_freqs": np.array([2, 4, 6]),
        "z_freqs": np.array([0]),
        "beta": 20,
        "min": 1e-1,
        "max": 1.0,
    }
    fourier_control = FourierControl("fourier_control", fourier_control_settings, fe_mesh)

    fe_mesh.Initialize()
    mechanical_loss_2d.Initialize()
    fourier_control.Initialize()

    # ============================================================
    # 2) load parametric samples and compute K(x,y)
    # ============================================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "fourier_control_dict.pkl")
    with open(file_path, "rb") as f:
        loaded_dict = pickle.load(f)

    coeffs_matrix = loaded_dict["coeffs_matrix"]
    # Original K from Fourier control (JAX array usually)
    K_matrix_jax = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

    # Work on a NumPy copy so we can edit sample 180
    K_matrix = np.array(K_matrix_jax)  # shape: (num_samples, n_nodes)

    num_samples, n_nodes = K_matrix.shape
    N = model_settings["N"]

    # ============================================================
    # 2.1) construct dual-phase K for sample 180 (circular inclusion)
    # ============================================================
    otf_id = 180
    if not (0 <= otf_id < num_samples):
        raise ValueError(f"otf_id={otf_id} out of range [0, {num_samples - 1}]")

    # reshape to (N, N) grid (same convention as plot_mesh_vec_data)
    K_base_180 = K_matrix[otf_id, :].copy()
    K_grid = K_base_180.reshape(N, N)

    # build coordinates for a simple "poly-like" dual phase:
    # central soft grain, outer hard matrix
    x_lin = np.linspace(0.0, model_settings["L"], N)
    y_lin = np.linspace(0.0, model_settings["L"], N)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

    cx, cy = 0.5, 0.5
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Soft / hard phase values
    K_soft = 0.01
    K_hard = 1.0

    # simple inclusion: r < 0.2 = soft, else hard
    mask_soft = r < 0.2
    K_dual_grid = np.where(mask_soft, K_soft, K_hard)

    # flatten back to vector
    K_dual_180 = K_dual_grid.reshape(-1)

    # ============================================================
    # 2.2) define control matrix for IdentityControl
    #      - all samples use original K
    #      - sample 180 is overwritten by dual-phase K
    # ============================================================
    K_control_matrix = K_matrix.copy()
    K_control_matrix[otf_id, :] = K_dual_180

    # ============================================================
    # 3) FNO model definition
    # ============================================================
    def merge_state(dst: nnx.State, src: nnx.State):
        for k, v in src.items():
            if isinstance(v, nnx.State):
                merge_state(dst[k], v)
            else:
                dst[k] = v

    fno_model = bridge.ToNNX(
        FourierNeuralOperator2D(
            modes1=6,
            modes2=6,
            width=8,
            depth=4,
            channels_last_proj=32,
            out_channels=2,
            output_scale=0.1,
        ),
        rngs=nnx.Rngs(0),
    ).lazy_init(K_control_matrix[0:1].reshape(1, N, N, 1))

    # replace RNG key by a dummy to allow checkpoint restoration
    graph_def, state = nnx.split(fno_model)
    rngs_key = jax.tree.map(jax.random.key_data, state.filter(nnx.RngKey))
    merge_state(state, rngs_key)
    fno_model = nnx.merge(graph_def, state)

    # count parameters
    params = nnx.state(fno_model, nnx.Param)
    total_params = sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(params))
    print(f"total number of fno network param: {total_params}")

    # use CLI value for epochs
    num_epochs = fol_num_epochs
    learning_rate_scheduler = optax.linear_schedule(
        init_value=1e-4, end_value=1e-5, transition_steps=num_epochs
    )
    optimizer = optax.chain(optax.adam(1e-3))

    # ============================================================
    # 4) IdentityControl over K(x,y)
    # ============================================================
    num_vars = n_nodes  # one K per node
    identity_control_settings = {"num_vars": num_vars}
    identity_control = IdentityControl("identity_control", identity_control_settings, fe_mesh)

    # ============================================================
    # 5) create FOL wrapper with IdentityControl
    # ============================================================
    pi_fno_pr_learning = PhysicsInformedFourierParametricOperatorLearning(
        name="pi_fno_pr_learning",
        control=identity_control,       # identity control on base mesh
        loss_function=mechanical_loss_2d,
        flax_neural_network=fno_model,
        optax_optimizer=optimizer,
    )

    pi_fno_pr_learning.Initialize()

    # ============================================================
    # 6) restore parametric weights (optional)
    # ============================================================
    if restore_param_init:
        # Load from a *previous* training run directory (parametric or OTF)
        param_case_dir = "./1_Parametric_reference_circular"
        param_state_dir = os.path.join(param_case_dir, "flax_train_state")
        print(f"[INFO] Restoring checkpoint from: {param_state_dir}")
        pi_fno_pr_learning.RestoreState(restore_state_directory=param_state_dir)
    else:
        print("[INFO] Skipping checkpoint restore. Using current (random) initialization.")

    # ============================================================
    # 7) Training setup: parametric or OTF-on-sample-180
    # ============================================================
    # eval_cases is always sample 180
    eval_cases = [otf_id]

    if not zssr_only:
        # Only define train/test sets if we are going to train
        if parametric_learning:
            # parametric training on many samples
            train_set = K_control_matrix[:180, :]
            test_set = K_control_matrix[181:200, :]
            batch_size = 5
            print("[INFO] Running PARAMETRIC training mode.")
        else:
            # only sample 180, dual-phase K (OTF)
            train_set = K_control_matrix[otf_id, :].reshape(1, -1)
            test_set = train_set.copy()
            batch_size = 1
            print("[INFO] Running OTF training on dual-phase sample 180.")

        # ========================================================
        # 8) Training (initialized from restored or random weights)
        # ========================================================
        pi_fno_pr_learning.Train(
            train_set=(train_set,),
            test_set=(test_set,),
            test_frequency=100,
            batch_size=batch_size,
            convergence_settings={
                "num_epochs": num_epochs,
                "relative_error": 1e-100,
                "absolute_error": 1e-100,
            },
            plot_settings={
                "plot_list": ["total_loss", "residual_rms_batch_mean"],
                "plot_frequency": 1,
                "save_frequency": 100,
                "save_directory": case_dir,
                "test_frequency": 100,
            },
            train_checkpoint_settings={
                "least_loss_checkpointing": True,
                "frequency": 100,
            },
            working_directory=case_dir,
        )

        # restore best checkpoint from THIS run
        pi_fno_pr_learning.RestoreState(
            restore_state_directory=os.path.join(case_dir, "flax_train_state")
        )
    else:
        print("[INFO] zssr_only=True -> skipping any additional training. Using restored weights only.")

    # ============================================================
    # 9) Base-resolution evaluation on dual-phase sample (180)
    #    Uses whatever weights are currently in pi_fno_pr_learning
    # ============================================================
    for eval_id in eval_cases:
        # ---------- FNO prediction (base 42x42) ----------
        K_input = K_control_matrix[eval_id, :].reshape(1, -1)
        FNO_UV = np.array(pi_fno_pr_learning.Predict(K_input)).reshape(-1)
        fe_mesh[f"U_FNO_{eval_id}"] = FNO_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

        if solve_FE:
            # ---------- FE solve with Newton residual tracking ----------
            fe_setting = {
                "linear_solver_settings": {
                    "solver": "JAX-bicgstab",
                    "tol": 1e-6,
                    "atol": 1e-6,
                    "maxiter": 1000,
                    "pre-conditioner": "ilu",
                },
                "nonlinear_solver_settings": {
                    "rel_tol": 1e-5,
                    "abs_tol": 1e-5,
                    "maxiter": 8,
                    "load_incr": 21,
                },
            }
            nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver(
                "nonlin_fe_solver", mechanical_loss_2d, fe_setting
            )
            nonlin_fe_solver.Initialize()

            ndofs = 2 * fe_mesh.GetNumberOfNodes()
            initial_dofs = np.zeros(ndofs)

            FE_UV_jax, residuals, total_iters = custom_newton_solve(
                fe_solver=nonlin_fe_solver,
                control_vars=K_control_matrix[eval_id, :],   # dual-phase K
                initial_dofs=initial_dofs,
                case_dir=case_dir,
                sample_tag=f"sample_{eval_id}",
            )
            print(f"Total Newton iterations (global) for sample {eval_id}: {total_iters}")

            FE_UV = np.array(FE_UV_jax)
            fe_mesh[f"U_FE_{eval_id}"] = FE_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

            # ---------- Newton residual convergence plot ----------
            iters = np.arange(1, len(residuals) + 1)
            plt.figure(figsize=(6, 4))
            plt.semilogy(iters, residuals, marker="o")
            plt.xlabel("Global Newton iteration")
            plt.ylabel(r"RMS residual $\|r\|_{\mathrm{rms}}$")
            plt.title(f"Newton convergence – dual-phase sample {eval_id}")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(case_dir, f"newton_residuals_sample_{eval_id}.png"))
            plt.close()

            # ---------- FNO–FE absolute error field ----------
            absolute_error = np.abs(FNO_UV.reshape(-1, 1) - FE_UV.reshape(-1, 1))
            fe_mesh[f"abs_error_{eval_id}"] = absolute_error.reshape(
                (fe_mesh.GetNumberOfNodes(), 2)
            )

            plot_mesh_vec_data(
                model_settings["L"],
                [
                    K_control_matrix[eval_id, :],  # heterogeneity (dual-phase for 180)
                    FNO_UV[::2],
                    FE_UV[::2],
                    absolute_error[::2],
                ],
                subplot_titles=["Heterogeneity", "FNO_U", "FE_U", "absolute_error"],
                fig_title=None,
                cmap="viridis",
                block_bool=True,
                colour_bar=True,
                colour_bar_name=None,
                X_axis_name=None,
                Y_axis_name=None,
                show=False,
                file_name=os.path.join(case_dir, f"plot_results_{eval_id}.png"),
            )
        else:
            print("[INFO] solve_FE=False, skipping FE and error computation for base resolution.")

    # ============================================================
    # 10) Optional ZSSR evaluation on higher resolution (e.g. 82x82)
    #      - Uses current FNO weights (restored + optional training)
    #      - Regenerates circular dual-phase K on new mesh
    #      - Runs FE on the new mesh and compares
    # ============================================================
    if do_zssr:
        print(f"[INFO] ZSSR mode ON. Evaluating at higher resolution N={zssr_N}.")

        # --- new mesh + loss for ZSSR resolution ---
        model_settings_z = model_settings.copy()
        model_settings_z["N"] = int(zssr_N)
        N_z = model_settings_z["N"]

        fe_mesh_z = create_2D_square_mesh(L=model_settings_z["L"], N=model_settings_z["N"])

        bc_dict_z = {
            "Ux": {
                "left": model_settings_z["Ux_left"],
                "right": model_settings_z["Ux_right"],
            },
            "Uy": {
                "left": model_settings_z["Uy_left"],
                "right": model_settings_z["Uy_right"],
            },
        }

        mechanical_loss_2d_z = NeoHookeMechanicalLoss2DQuad(
            "mechanical_loss_2d_zssr",
            loss_settings={
                "dirichlet_bc_dict": bc_dict_z,
                "num_gp": 2,
                "material_dict": material_dict,
            },
            fe_mesh=fe_mesh_z,
        )

        fe_mesh_z.Initialize()
        mechanical_loss_2d_z.Initialize()

        # --- dual-phase K at higher resolution (same circle, new grid) ---
        x_lin_z = np.linspace(0.0, model_settings_z["L"], N_z)
        y_lin_z = np.linspace(0.0, model_settings_z["L"], N_z)
        X_z, Y_z = np.meshgrid(x_lin_z, y_lin_z, indexing="ij")

        cx, cy = 0.5, 0.5
        r_z = np.sqrt((X_z - cx) ** 2 + (Y_z - cy) ** 2)

        mask_soft_z = r_z < 0.2
        K_dual_grid_z = np.where(mask_soft_z, K_soft, K_hard)
        K_dual_z = K_dual_grid_z.reshape(-1)

        # store heterogeneity on ZSSR mesh
        fe_mesh_z[f"K_dual_zssr_{otf_id}"] = K_dual_z

        # --- FNO prediction at higher resolution (direct call) ---
        # FNO is resolution-agnostic w.r.t. spatial grid, so we pass the new (N_z, N_z, 1) field.
        K_input_z = K_dual_z.reshape(1, N_z, N_z, 1)
        FNO_UV_z = np.array(fno_model(K_input_z)).reshape(-1)
        fe_mesh_z[f"U_FNO_zssr_{otf_id}"] = FNO_UV_z.reshape(
            (fe_mesh_z.GetNumberOfNodes(), 2)
        )

        if solve_FE:
            # --- FE solve on ZSSR mesh ---
            fe_setting_z = {
                "linear_solver_settings": {
                    "solver": "JAX-bicgstab",
                    "tol": 1e-6,
                    "atol": 1e-6,
                    "maxiter": 1000,
                    "pre-conditioner": "ilu",
                },
                "nonlinear_solver_settings": {
                    "rel_tol": 1e-7,
                    "abs_tol": 1e-7,
                    "maxiter": 8,
                    "load_incr": 10,
                },
            }

            nonlin_fe_solver_z = FiniteElementNonLinearResidualBasedSolver(
                "nonlin_fe_solver_zssr", mechanical_loss_2d_z, fe_setting_z
            )
            nonlin_fe_solver_z.Initialize()

            ndofs_z = 2 * fe_mesh_z.GetNumberOfNodes()
            initial_dofs_z = np.zeros(ndofs_z)

            FE_UV_jax_z, residuals_z, total_iters_z = custom_newton_solve(
                fe_solver=nonlin_fe_solver_z,
                control_vars=K_dual_z,
                initial_dofs=initial_dofs_z,
                case_dir=case_dir,
                sample_tag=f"zssr_sample_{otf_id}_N{N_z}",
            )

            print(
                f"Total Newton iterations (global) for ZSSR sample {otf_id} at N={N_z}: "
                f"{total_iters_z}"
            )

            FE_UV_z = np.array(FE_UV_jax_z)
            fe_mesh_z[f"U_FE_zssr_{otf_id}"] = FE_UV_z.reshape(
                (fe_mesh_z.GetNumberOfNodes(), 2)
            )

            # --- ZSSR Newton residual convergence plot ---
            iters_z = np.arange(1, len(residuals_z) + 1)
            plt.figure(figsize=(6, 4))
            plt.semilogy(iters_z, residuals_z, marker="o")
            plt.xlabel("Global Newton iteration")
            plt.ylabel(r"RMS residual $\|r\|_{\mathrm{rms}}$")
            plt.title(f"Newton convergence – ZSSR dual-phase N={N_z} sample {otf_id}")
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    case_dir,
                    f"newton_residuals_zssr_sample_{otf_id}_N{N_z}.png",
                )
            )
            plt.close()

            # --- ZSSR FNO–FE absolute error field ---
            absolute_error_z = np.abs(FNO_UV_z.reshape(-1, 1) - FE_UV_z.reshape(-1, 1))
            fe_mesh_z[f"abs_error_zssr_{otf_id}"] = absolute_error_z.reshape(
                (fe_mesh_z.GetNumberOfNodes(), 2)
            )

            plot_mesh_vec_data(
                model_settings_z["L"],
                [
                    K_dual_z,
                    FNO_UV_z[::2],
                    FE_UV_z[::2],
                    absolute_error_z[::2],
                ],
                subplot_titles=[
                    f"Heterogeneity ZSSR N={N_z}",
                    f"FNO_Ux ZSSR N={N_z}",
                    f"FE_Ux ZSSR N={N_z}",
                    "absolute_error ZSSR",
                ],
                fig_title=None,
                cmap="viridis",
                block_bool=True,
                colour_bar=True,
                colour_bar_name=None,
                X_axis_name=None,
                Y_axis_name=None,
                show=False,
                file_name=os.path.join(
                    case_dir,
                    f"zssr_results_sample_{otf_id}_N{N_z}.png",
                ),
            )
        else:
            print("[INFO] solve_FE=False, skipping FE and error computation for ZSSR resolution.")

        # --- Export ZSSR mesh to its own subfolder ---
        zssr_dir = os.path.join(case_dir, f"ZSSR_N{N_z}")
        os.makedirs(zssr_dir, exist_ok=True)
        fe_mesh_z.Finalize(export_dir=zssr_dir)

    # ============================================================
    # 11) Export base mesh and optional cleanup
    # ============================================================
    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)


if __name__ == "__main__":
    # defaults
    fol_num_epochs = 10000
    solve_FE = True
    clean_dir = False
    restore_param_init = True
    parametric_learning = False
    do_zssr = True
    zssr_N = 82
    zssr_only = True  # default here: ZSSR-only using restored weights

    args = sys.argv[1:]
    for arg in args:
        if arg.startswith("fol_num_epochs="):
            try:
                fol_num_epochs = int(arg.split("=")[1])
            except ValueError:
                print("fol_num_epochs should be an integer.")
                sys.exit(1)
        elif arg.startswith("solve_FE="):
            value = arg.split("=")[1]
            if value.lower() in ["true", "false"]:
                solve_FE = value.lower() == "true"
            else:
                print("solve_FE should be True or False.")
                sys.exit(1)
        elif arg.startswith("clean_dir="):
            value = arg.split("=")[1]
            if value.lower() in ["true", "false"]:
                clean_dir = value.lower() == "true"
            else:
                print("clean_dir should be True or False.")
                sys.exit(1)
        elif arg.startswith("restore_param_init="):
            value = arg.split("=")[1]
            if value.lower() in ["true", "false"]:
                restore_param_init = value.lower() == "true"
            else:
                print("restore_param_init should be True or False.")
                sys.exit(1)
        elif arg.startswith("parametric_learning="):
            value = arg.split("=")[1]
            if value.lower() in ["true", "false"]:
                parametric_learning = value.lower() == "true"
            else:
                print("parametric_learning should be True or False.")
                sys.exit(1)
        elif arg.startswith("do_zssr="):
            value = arg.split("=")[1]
            if value.lower() in ["true", "false"]:
                do_zssr = value.lower() == "true"
            else:
                print("do_zssr should be True or False.")
                sys.exit(1)
        elif arg.startswith("zssr_N="):
            try:
                zssr_N = int(arg.split("=")[1])
            except ValueError:
                print("zssr_N should be an integer.")
                sys.exit(1)
        elif arg.startswith("zssr_only="):
            value = arg.split("=")[1]
            if value.lower() in ["true", "false"]:
                zssr_only = value.lower() == "true"
            else:
                print("zssr_only should be True or False.")
                sys.exit(1)
        else:
            print(
                "Usage: python reference_clean_otf_zssr_circular.py "
                "fol_num_epochs=10000 solve_FE=True clean_dir=False "
                "restore_param_init=True parametric_learning=False "
                "do_zssr=True zssr_N=82 zssr_only=True"
            )
            sys.exit(1)

    main(
        fol_num_epochs,
        solve_FE,
        clean_dir,
        restore_param_init,
        parametric_learning,
        do_zssr,
        zssr_N,
        zssr_only,
    )
