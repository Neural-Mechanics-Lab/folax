#!/usr/bin/env python3

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import pickle
import numpy as np
import optax
from pathlib import Path

from fol.loss_functions.regression_loss import RegressionLoss
from fol.controls.identity_control      import IdentityControl

from data_driven_meta_implicit_parametric_operator_learning_3D import (
    DataDrivenMetaImplicitParametricOperatorLearning)
from fol.deep_neural_networks.nns     import HyperNetwork, MLP
from fol.mesh_input_output.mesh       import Mesh
from fol.tools.usefull_functions      import *
from fol.tools.logging_functions      import Logger

# ── directory & logging ──────────────────────────────────────────────────
working_directory_name = "implicit_operator_learning_3d"
case_dir = os.path.join(".", working_directory_name)
create_clean_directory(working_directory_name)
sys.stdout = Logger(os.path.join(case_dir, working_directory_name + ".log"))

# ── dataset & mesh paths (unchanged) ─────────────────────────────────────
PKL_PATH  = "/home/jerry/200_NIN_RESULTS/gt_values.pkl"
MESH_FILE = "/home/jerry/50_NIN_RESULTS/examples/paper_NIN/example_3d/cylindrical_fine_all_sides.med"

with open(PKL_PATH, "rb") as f:
    pkl = pickle.load(f)

keys        = sorted(pkl.keys())
X_list      = [pkl[k]["coeffs_matrix"] for k in keys]          # (3,)
U_FEM_list  = [pkl[k]["FE_UVW"]        for k in keys]          # (3*nodes,)

data_sets = {
    "X"    : np.vstack(X_list ).astype(np.float32),
    "U_FEM": np.vstack(U_FEM_list).astype(np.float32),
}

print(f"[INFO] Loaded {len(data_sets['X'])} samples – displacement DOF = {data_sets['U_FEM'].shape[1]}")

# ── problem setup ────────────────────────────────────────────────────────
fe_mesh = Mesh("fol_io", Path(MESH_FILE).name, str(Path(MESH_FILE).parent))
fe_mesh.Initialize()

identity_control = IdentityControl("ident_control", num_vars=3)

reg_loss = RegressionLoss(
    "reg_loss",
    loss_settings={"nodal_unknows": ["Ux", "Uy", "Uz"]},
    fe_mesh=fe_mesh,
)
reg_loss.Initialize()
identity_control.Initialize()

# ── FiLM-SIREN definition  ───────────────────────────────────
characteristic_length = 64
synthesizer_nn = MLP(
    name="regressor_synthesizer",
    input_size=3,
    output_size=3,
    hidden_layers=[characteristic_length] * 6,
    activation_settings={
        "type": "sin",
        "prediction_gain": 30,
        "initialization_gain": 1.0,
    },
)
latent_size  = 10
modulator_nn = MLP(name="modulator_nn", input_size=latent_size, use_bias=False)

hyper_network = HyperNetwork(
    name="hyper_nn",
    modulator_nn=modulator_nn,
    synthesizer_nn=synthesizer_nn,
    coupling_settings={"modulator_to_synthesizer_coupling_mode": "one_modulator_per_synthesizer_layer"},
)

# ── single optimiser (no latent loop - no second optimiser) ─────────────
num_epochs = 10000
main_loop_transform = optax.chain(
    optax.normalize_by_update_norm(),
    optax.adam(1e-5),
)

# ── learner (no latent-step args needed) ─────────────────────────────────
fol = DataDrivenMetaImplicitParametricOperatorLearning(
    name                      ="implicit_ol_3d",
    control                   = identity_control,
    loss_function             = reg_loss,
    flax_neural_network       = hyper_network,
    main_loop_optax_optimizer = main_loop_transform,
    num_latent_iterations     = 0,  # no latent loop
)

fol.Initialize()

# ── manual train / test / eval split via blocks ──────────────────────────
N = len(data_sets["X"])

train_blocks = [(50, 100), (130, 160)]    #mixing for better training
eval_blocks  = [(100, 105)]              
test_blocks  = [(105, 130), (160, 200)]   

def blocks_to_idx(blocks):
    return np.concatenate([np.arange(s, min(e, N)) for s, e in blocks if s < e])

idx_train = np.unique(blocks_to_idx(train_blocks))
idx_eval  = np.unique(blocks_to_idx(eval_blocks))
idx_test  = np.unique(blocks_to_idx(test_blocks))

# sanity: sets must be pair-wise disjoint
if len(np.intersect1d(idx_train, idx_eval)) \
or len(np.intersect1d(idx_train, idx_test)) \
or len(np.intersect1d(idx_eval , idx_test)):
    raise ValueError("train / eval / test index ranges overlap!")

print(f"[INFO] Train {idx_train.shape[0]}  Eval {idx_eval.shape[0]}  "
      f"Test {idx_test.shape[0]} samples  (N = {N})")

# ── training call ───────────────────────────────────────────────────────
fol.Train(
    train_set=(data_sets["X"][idx_train, :],
               data_sets["U_FEM"][idx_train, :]),
    test_set =(data_sets["X"][idx_test , :],
               data_sets["U_FEM"][idx_test , :]),
    test_frequency = 10,
    batch_size     = 5,
    convergence_settings      = {"num_epochs": num_epochs,
                                 "relative_error": 1e-100,
                                 "absolute_error": 1e-100},
    train_checkpoint_settings = {"least_loss_checkpointing": True,
                                 "frequency": 10},
    working_directory         = case_dir,
)

fol.RestoreState(restore_state_directory=f"{case_dir}/flax_train_state")

# ── evaluation & VTU export (one file per sample) ───────────────────────
EXPORT_VTU = True
out_dir = os.path.join(case_dir, "tested_samples")
os.makedirs(out_dir, exist_ok=True)

for k in idx_eval:                                   # iterate eval samples
    U_pred_flat = np.array(
        fol.Predict(data_sets["X"][k][None, :])
    ).reshape(-1)

    U_true_flat = data_sets["U_FEM"][k]
    U_pred      = U_pred_flat.reshape(-1, 3)
    U_true      = U_true_flat.reshape(-1, 3)
    abs_err_mag = np.linalg.norm(U_pred - U_true, axis=1)

    if EXPORT_VTU:
        m = Mesh("fol_io", Path(MESH_FILE).name, str(Path(MESH_FILE).parent))
        m.Initialize()
        m["U_FE"]    = U_true.astype(np.float32)
        m["U_pred"]  = U_pred.astype(np.float32)
        m["abs_err"] = abs_err_mag.astype(np.float32)
        m.file_name  = f"sample_{k}.vtu"
        m.Finalize(export_dir=out_dir, export_format="vtu")
        print(f"   VTU field written to {m.file_name}")

print(" all VTU files saved in", out_dir)
