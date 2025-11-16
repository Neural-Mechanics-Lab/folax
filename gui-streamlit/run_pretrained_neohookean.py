import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss2DQuad
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from identity_control import IdentityControl
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.usefull_functions import *
from fol.tools.logging_functions import *
from fol.deep_neural_networks.nns import HyperNetwork, MLP
import optax
from flax import nnx
from mechanical2d_utilities import *
from fol.tools.decoration_functions import *


def parse_args():
    """Parse sys.argv for N (grid size)."""
    N = 40  # default
    for arg in sys.argv[1:]:
        if arg.startswith("N="):
            N = int(arg.split("=")[1])
    return N


def plot_deformation_with_microstructure(fe_mesh, u, K_matrix, ax=None, scale=1.0, cmap=None):
    coords = np.asarray(fe_mesh.nodes_coordinates)[:, :2].copy()
    disp = np.asarray(u).reshape(-1, 2)
    values = np.asarray(K_matrix)
    coords[:, 1] = coords[:, 1].max() - coords[:, 1]

    # coords_def = coords + scale * disp
    coords_def = coords + disp

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
        fig, ax = plt.subplots(figsize=(6, 6))
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


def main():
    # Parse arguments
    N = parse_args()
    L = 1

    # Define mesh
    fe_mesh = create_2D_square_mesh(L=L, N=N)
    fe_mesh.Initialize()

    # Boundary conditions & material
    bc_dict = {"Ux": {"left": 0.0, "right": 0.5}, "Uy": {"left": 0.0, "right": 0.5}}
    material_dict = {"young_modulus": 1, "poisson_ratio": 0.3}

    mechanical_loss_2d = NeoHookeMechanicalLoss2DQuad(
        "mechanical_loss_2d",
        loss_settings={"dirichlet_bc_dict": bc_dict, "num_gp": 2, "material_dict": material_dict},
        fe_mesh=fe_mesh,
    )
    mechanical_loss_2d.Initialize()

    identity_control = IdentityControl("identity_control", control_settings={}, fe_mesh=fe_mesh)
    identity_control.Initialize

    # Load K_matrix
    K_matrix_file = os.path.join(os.getcwd(), "K_matrix.npy")
    if not os.path.exists(K_matrix_file):
        raise FileNotFoundError("K_matrix.npy not found in current directory.")
    K_matrix = np.load(K_matrix_file)

    # Define pretrained model architecture
    synthesizer_nn = MLP(
        name="synthesizer_nn",
        input_size=3,
        output_size=2,
        hidden_layers=[64] * 4,
        activation_settings={"type": "sin", "prediction_gain": 30, "initialization_gain": 1.0},
        skip_connections_settings={"active": False, "frequency": 1},
    )
    modulator_nn = MLP(name="modulator_nn", input_size=8 * 64, use_bias=False)
    hyper_network = HyperNetwork(
        name="hyper_nn",
        modulator_nn=modulator_nn,
        synthesizer_nn=synthesizer_nn,
        coupling_settings={"modulator_to_synthesizer_coupling_mode": "one_modulator_per_synthesizer_layer"},
    )

    # Optimizers (dummy, for restoring)
    main_loop_transform = optax.chain(optax.adam(1e-5))
    latent_step_optimizer = optax.chain(optax.adam(1e-4))

    ifol = MetaAlphaMetaImplicitParametricOperatorLearning(
        name="meta_implicit_fol",
        control=identity_control,
        loss_function=mechanical_loss_2d,
        flax_neural_network=hyper_network,
        main_loop_optax_optimizer=main_loop_transform,
        latent_step_optax_optimizer=latent_step_optimizer,
        latent_step_size=1e-2,
        num_latent_iterations=3,
    )
    ifol.Initialize()

    # Restore pretrained weights
    case_dir = "./mechanical_2d_base_from_ifol_meta"
    ifol.RestoreState(restore_state_directory=os.path.join(case_dir, "flax_train_state"))

    # Predict with pretrained model
    iFOL_UVW = np.array(ifol.Predict(K_matrix))

    for eval_id in range(K_matrix.shape[0]):
        ifol_uvw = iFOL_UVW[eval_id, :]
        fe_mesh[f"iFOL_U_{eval_id}"] = ifol_uvw.reshape((fe_mesh.GetNumberOfNodes(), 2))

    abs_errors = []

    # FE baseline solve for comparison
    for eval_id in range(K_matrix.shape[0]):
        K_sample = K_matrix[eval_id, :]

        fe_setting = {
            "linear_solver_settings": {"solver": "JAX-direct"},
            "nonlinear_solver_settings": {"rel_tol": 1e-8, "abs_tol": 1e-8, "maxiter": 8, "load_incr": 40},
        }
        nonlin_fe_solver = FiniteElementNonLinearResidualBasedSolver("nonlin_fe_solver", mechanical_loss_2d, fe_setting)
        nonlin_fe_solver.Initialize()

        try:
            FE_UVW = np.array(nonlin_fe_solver.Solve(K_sample, np.zeros(2 * fe_mesh.GetNumberOfNodes())))
        except:
            FE_UVW = np.zeros(2 * fe_mesh.GetNumberOfNodes())

        ifol_uvw = iFOL_UVW[eval_id, :]
        abs_err = np.abs(FE_UVW - ifol_uvw)
        abs_errors.append(abs_err)

        fe_mesh[f"FE_U_{eval_id}"] = FE_UVW.reshape((fe_mesh.GetNumberOfNodes(), 2))
        fe_mesh[f"abs_U_error_{eval_id}"] = abs_err.reshape((fe_mesh.GetNumberOfNodes(), 2))

    # Export results
    fe_mesh.Finalize(export_dir=case_dir)

    # Plots
    eval_id = 0
    Ux_2d = iFOL_UVW[eval_id].reshape(N, N, 2)[:, :, 0]
    Uy_2d = iFOL_UVW[eval_id].reshape(N, N, 2)[:, :, 1]
    abs_U_2d = abs_errors[eval_id].reshape(N, N, 2)[:, :, 0]
    abs_V_2d = abs_errors[eval_id].reshape(N, N, 2)[:, :, 1]

    plot_mesh_vec_data(
        1,
        [Ux_2d, Uy_2d, abs_U_2d, abs_V_2d],
        ["U", "V", "abs_error_U", "abs_error_V"],
        fig_title="implicit FOL solution and error",
        file_name=os.path.join(case_dir, f"FOL-UV-dist_test_{eval_id}.png"),
    )

    # Side-by-side FE vs FOL deformation
    K_matrix_nodes = np.ravel(K_matrix[eval_id, :])
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_deformation_with_microstructure(fe_mesh, FE_UVW, K_matrix_nodes, ax=axes[0], scale=5.0)
    axes[0].set_title("FE Deformation")

    plot_deformation_with_microstructure(fe_mesh, iFOL_UVW, K_matrix_nodes, ax=axes[1], scale=5.0)
    axes[1].set_title("FOL Deformation")

    plt.tight_layout()
    plt.savefig(os.path.join(case_dir, f"FE_FOL_deformation_{eval_id}.png"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
