import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt

from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import (
    FiniteElementNonLinearResidualBasedSolver,
)
from fol.controls.fourier_control import FourierControl
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from newton_residual_tracker import custom_newton_solve
import pickle


def main(sample_id=180, clean_dir=False):
    # ============================================================
    # 0) directory & logging
    # ============================================================
    working_directory_name = "FE_only_Transfer_sample_180"
    case_dir = os.path.join(".", working_directory_name)
    create_clean_directory(working_directory_name)
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

    # creation of the model
    fe_mesh = create_2D_square_mesh(L=model_settings["L"], N=model_settings["N"])

    # FE loss (Neo-Hooke)
    bc_dict = {
        "Ux": {"left": model_settings["Ux_left"], "right": model_settings["Ux_right"]},
        "Uy": {"left": model_settings["Uy_left"], "right": model_settings["Uy_right"]},
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

    # Fourier control (for generating heterogeneity K(x,y))
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
    K_matrix = fourier_control.ComputeBatchControlledVariables(coeffs_matrix)

    num_samples = coeffs_matrix.shape[0]
    if not (0 <= sample_id < num_samples):
        raise ValueError(
            f"sample_id={sample_id} is out of range [0, {num_samples - 1}]"
        )

    # this is exactly the K-field used in your original script for FE
    K_used = np.array(K_matrix[sample_id, :])
    fe_mesh[f"K_fourier_{sample_id}"] = K_used

    print(
        f"[INFO] FE-only run for sample {sample_id} with Fourier K-field "
        f"(N={model_settings['N']}, L={model_settings['L']})"
    )

    # ============================================================
    # 3) FE settings
    #    (same as in the original TransferLearn script)
    # ============================================================
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

    # ============================================================
    # 4) Newton solve (baseline from zero DOFs)
    # ============================================================
    ndofs = 2 * fe_mesh.GetNumberOfNodes()
    initial_dofs = np.zeros(ndofs)

    FE_UV_jax, residuals, total_iters = custom_newton_solve(
        fe_solver=nonlin_fe_solver,
        control_vars=K_used,
        initial_dofs=initial_dofs,
        case_dir=case_dir,
        sample_tag=f"sample_{sample_id}",
    )

    print(f"Total Newton iterations (global) for sample {sample_id}: {total_iters}")

    FE_UV = np.array(FE_UV_jax)
    fe_mesh[f"U_FE_only_{sample_id}"] = FE_UV.reshape(
        (fe_mesh.GetNumberOfNodes(), 2)
    )

    # ============================================================
    # 5) Newton residual convergence plot
    # ============================================================
    iters = np.arange(1, len(residuals) + 1)
    plt.figure(figsize=(6, 4))
    plt.semilogy(iters, residuals, marker="o")
    plt.xlabel("Global Newton iteration")
    plt.ylabel(r"RMS residual $\|r\|_{\mathrm{rms}}$")
    plt.title(f"Newton convergence – FE-only sample {sample_id}")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, f"newton_residuals_FE_only_sample_{sample_id}.png"))
    plt.close()

    # ============================================================
    # 6) Plot heterogeneity + FE Ux
    # ============================================================
    plot_mesh_vec_data(
        model_settings["L"],
        [K_used, FE_UV[::2]],
        subplot_titles=["Fourier heterogeneity", "FE_Ux (FE-only)"],
        fig_title=None,
        cmap="viridis",
        block_bool=True,
        colour_bar=True,
        colour_bar_name=None,
        X_axis_name=None,
        Y_axis_name=None,
        show=False,
        file_name=os.path.join(
            case_dir, f"fe_only_results_sample_{sample_id}.png"
        ),
    )

    # ============================================================
    # 7) Export mesh
    # ============================================================
    fe_mesh.Finalize(export_dir=case_dir)

    if clean_dir:
        shutil.rmtree(case_dir)


if __name__ == "__main__":
    sample_id = 180
    clean_dir = False

    args = sys.argv[1:]
    for arg in args:
        if arg.startswith("sample_id="):
            try:
                sample_id = int(arg.split("=")[1])
            except ValueError:
                print("sample_id should be an integer.")
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
                "Usage: python FE_only_Transfer_sample_180.py "
                "sample_id=180 clean_dir=False"
            )
            sys.exit(1)

    main(sample_id=sample_id, clean_dir=clean_dir)
