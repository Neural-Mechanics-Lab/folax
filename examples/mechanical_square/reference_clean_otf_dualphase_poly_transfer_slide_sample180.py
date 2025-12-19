import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
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


def main(fol_num_epochs=10, solve_FE=False, clean_dir=False):
    # ============================================================
    # 0) directory & logging
    # ============================================================
    working_directory_name = 'nn_output_mechanical_2D_neohooke_pi_fno_otf_dualphase_slide_sample180'
    case_dir = os.path.join('.', working_directory_name)
    create_clean_directory(working_directory_name)
    os.makedirs(case_dir, exist_ok=True)  # safety
    sys.stdout = Logger(os.path.join(case_dir, working_directory_name + ".log"))

    # ============================================================
    # 1) problem setup
    # ============================================================
    model_settings = {
        "L": 1.0,
        "N": 42,
        "Ux_left": 0.0,
        "Ux_right": 0.1,
        "Uy_left": 0.0,
        "Uy_right": 0.1,
    }

    # mesh
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
    # 2.1) construct dual-phase K for sample 180
    #      (same threshold logic as FE-only dual-phase script)
    # ============================================================
    otf_id = 180
    if not (0 <= otf_id < num_samples):
        raise ValueError(f"otf_id={otf_id} out of range [0, {num_samples - 1}]")

    # continuous K from Fourier for this sample
    K_base_180 = K_matrix[otf_id, :].copy()

    # two phase levels (relative stiffness)
    K_soft = 0.01   # soft phase
    K_hard = 1.0    # hard phase

    # threshold at mid of min/max of original K
    thr = 0.5 * (K_base_180.min() + K_base_180.max())

    # dual-phase field: high values -> hard, low values -> soft
    K_dual_180 = np.where(K_base_180 >= thr, K_hard, K_soft)

    print(
        f"[INFO] Dual-phase K for sample {otf_id}: "
        f"K_soft={K_soft}, K_hard={K_hard}, thr={thr:.4f}"
    )

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
    print(f"total number of fno network param:{total_params}")

    num_epochs = 5000
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
        control=identity_control,       # <--- identity control now
        loss_function=mechanical_loss_2d,
        flax_neural_network=fno_model,
        optax_optimizer=optimizer,
    )

    pi_fno_pr_learning.Initialize()

    # ============================================================
    # 6) restore parametric weights from slide 2
    # ============================================================
    param_case_dir = "./nn_output_mechanical_2D_neohooke_pi_fno_param"
    param_state_dir = os.path.join(param_case_dir, "flax_train_state")
    print(f"[INFO] Restoring parametric checkpoint from: {param_state_dir}")
    pi_fno_pr_learning.RestoreState(restore_state_directory=param_state_dir)

    # ============================================================
    # 7) OTF setup: fine-tune only on dual-phase sample 180
    # ============================================================
    train_start_id = 0
    train_end_id = 180
    test_start_id = 181
    test_end_id = 200

    parametric_learning = False  # OTF only
    if parametric_learning:
        train_set = K_control_matrix[train_start_id:train_end_id, :]
        test_set = K_control_matrix[test_start_id:test_end_id, :]
        eval_cases = [otf_id]
        batch_size = 5
    else:
        # only sample 180, dual-phase K
        train_set = K_control_matrix[otf_id, :].reshape(1, -1)
        test_set = train_set.copy()
        eval_cases = [otf_id]
        batch_size = 1

    # ============================================================
    # 8) OTF training (initialized from parametric weights)
    # ============================================================
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
            "save_directory": case_dir,   # save plots into slide-4 folder
            "test_frequency": 100,
        },
        train_checkpoint_settings={
            "least_loss_checkpointing": True,
            "frequency": 100,
        },
        working_directory=case_dir,
    )

    # restore best OTF checkpoint from THIS run
    pi_fno_pr_learning.RestoreState(
        restore_state_directory=os.path.join(case_dir, "flax_train_state")
    )

    # ============================================================
    # 9) Evaluation on the same dual-phase OTF sample (180)
    # ============================================================
    for eval_id in eval_cases:
        # ---------- FNO prediction ----------
        K_input = K_control_matrix[eval_id, :].reshape(1, -1)
        FNO_UV = np.array(pi_fno_pr_learning.Predict(K_input)).reshape(-1)
        fe_mesh[f"U_FNO_{eval_id}"] = FNO_UV.reshape((fe_mesh.GetNumberOfNodes(), 2))

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

    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)


if __name__ == "__main__":
    fol_num_epochs = 2000
    solve_FE = False
    clean_dir = False

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
        else:
            print(
                "Usage: python reference_clean_otf_dualphase_poly_transfer_slide_4.py "
                "fol_num_epochs=2000 solve_FE=False clean_dir=False"
            )
            sys.exit(1)

    main(fol_num_epochs, solve_FE, clean_dir)
