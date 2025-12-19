import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import matplotlib.pyplot as plt

from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import (
    FiniteElementNonLinearResidualBasedSolver,
)
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import Logger
from newton_residual_tracker import custom_newton_solve


def main(sample_id=180, clean_dir=False):
    # ───────────────────── directory & logging ─────────────────────
    working_directory_name = "FE_only_circular_82x82"
    case_dir = os.path.join(".", working_directory_name)
    create_clean_directory(working_directory_name)
    sys.stdout = Logger(os.path.join(case_dir, working_directory_name + ".log"))

    # ───────────────────── problem setup ─────────────────────
    model_settings = {
        "L": 1.0,
        "N": 82,           # match your circular script
        "Ux_left": 0.0,
        "Ux_right": 0.1,
        "Uy_left": 0.0,
        "Uy_right": 0.1,
    }

    L = float(model_settings["L"])
    N = model_settings["N"]

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

    fe_mesh.Initialize()
    mechanical_loss_2d.Initialize()

    # ───────────────────── circular dual-phase K ─────────────────────
    x_lin = np.linspace(0.0, L, N)
    y_lin = np.linspace(0.0, L, N)
    X, Y = np.meshgrid(x_lin, y_lin, indexing="ij")

    cx, cy = 0.5 * L, 0.5 * L
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    K_soft = 0.01    # match your script
    K_hard = 1.0
    radius = 0.2 * L

    K_circ_grid = np.where(r < radius, K_soft, K_hard)
    K_used = K_circ_grid.reshape(-1)

    fe_mesh[f"K_circular_{sample_id}"] = K_used

    print(
        f"[INFO] Using circular dual-phase K for sample {sample_id}: "
        f"K_soft={K_soft}, K_hard={K_hard}, radius={radius:.3f}, N={N}"
    )

    # ───────────────────── FE settings ─────────────────────
    fe_setting = {
        "nonlinear_solver_settings": {
            "rel_tol": 1e-5,
            "abs_tol": 1e-5,
            "maxiter": 8,
            "load_incr": 21,   # match your circular script
        },
    }

    nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver(
        "nonlin_fe_solver", mechanical_loss_2d, fe_setting
    )
    nonlin_fe_solver.Initialize()

    # ───────────────────── baseline Newton from zero DOFs ─────────────────────
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
    fe_mesh[f"U_FE_circular_{sample_id}"] = FE_UV.reshape(
        (fe_mesh.GetNumberOfNodes(), 2)
    )

    # ───────────────────── Newton residual convergence plot ─────────────────────
    iters = np.arange(1, len(residuals) + 1)
    plt.figure(figsize=(6, 4))
    plt.semilogy(iters, residuals, marker="o")
    plt.xlabel("Global Newton iteration")
    plt.ylabel(r"RMS residual $\|r\|_{\mathrm{rms}}$")
    plt.title(f"Newton convergence – circular inclusion sample {sample_id}")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, f"newton_residuals_circular_sample_{sample_id}.png"))
    plt.close()

    # ───────────────────── plot: heterogeneity + FE Ux ─────────────────────
    plot_mesh_vec_data(
        model_settings["L"],
        [K_used, FE_UV[::2]],
        subplot_titles=["Circular dual-phase heterogeneity", "FE_Ux (circular inclusion)"],
        fig_title=None,
        cmap="viridis",
        block_bool=True,
        colour_bar=True,
        colour_bar_name=None,
        X_axis_name=None,
        Y_axis_name=None,
        show=False,
        file_name=os.path.join(
            case_dir, f"fe_only_circular_results_sample_{sample_id}.png"
        ),
    )

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
                "Usage: python mechanical_2D_neohooke_FE_only_circular_82x82.py "
                "sample_id=180 clean_dir=False"
            )
            sys.exit(1)

    main(sample_id=sample_id, clean_dir=clean_dir)
