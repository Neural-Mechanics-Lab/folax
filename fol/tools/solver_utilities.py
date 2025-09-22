import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..','..')))
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax.numpy as jnp
from fol.loss_functions.mechanical_neohooke import NeoHookeMechanicalLoss3DTetra
from fol.solvers.fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.mesh_input_output.mesh import Mesh
from fol.tools.usefull_functions import create_clean_directory
from fol.tools.logging_functions import Logger
import shutil
import time
# ───────────────────────────────────────────────── Residual tracker
class ResidualTracker:
    def __init__(self, case_dir, sample_id=0):
        self.csv = os.path.join(case_dir, f"residuals_sample_{sample_id}.csv")
        with open(self.csv, "w") as f:
            f.write("load_step,newton_iter,residual\n")
    def log(self, load_step, newton_iter, res_norm):
        val = float(res_norm)
        print(f"[NH] step {load_step:03d}  it {newton_iter:02d}  ||r||={val:.3e}")
        with open(self.csv, "a") as f:
            f.write(f"{load_step},{newton_iter},{val:.8e}\n")

# ───────────────────────────────────────────────── Newton + rollback + hard stop
def custom_solve(
    fe_solver,
    control_vars,
    initial_dofs,
    case_dir,
    sample_id,
    target_best=1e-6,   # stop early if we ever beat this
    growth_tol=50.0     # if residual grows > growth_tol×prev, abort
):
    """
    FE Newton with a divergence guard:
      • track best residual & DOFs
      • if residual becomes non-finite OR jumps too much → rollback to best and STOP
      • else run up to maxiter per load step
    """
    tracker = ResidualTracker(case_dir, sample_id)

    residuals = []
    applied_dofs = initial_dofs
    n_incr = fe_solver.nonlinear_solver_settings["load_incr"]
    maxit  = fe_solver.nonlinear_solver_settings["maxiter"]
    atol   = fe_solver.nonlinear_solver_settings["abs_tol"]
    rtol   = fe_solver.nonlinear_solver_settings["rel_tol"]

    # best-so-far snapshot
    best_res  = np.inf
    best_step = (0, 0)  # (load_step, iter)
    best_dofs = jnp.array(applied_dofs)

    for fac in range(n_incr):
        # ramp BCs
        applied_dofs = fe_solver.fe_loss_function.ApplyDirichletBCOnDofVector(
            applied_dofs, (fac + 1) / n_incr
        )

        prev_res = None
        for it in range(1, maxit + 1):
            jac, res = fe_solver.fe_loss_function.ComputeJacobianMatrixAndResidualVector(
                control_vars, applied_dofs
            )
            res_norm = float(jnp.linalg.norm(res, ord=2))
            residuals.append(res_norm)
            tracker.log(load_step=fac + 1, newton_iter=it, res_norm=res_norm)

            # Non-finite? → rollback & STOP
            if not np.isfinite(res_norm):
                print(f"[NH-guard] Non-finite residual at step {fac+1}, it {it}. "
                      f"Rolling back to best (||r||={best_res:.3e}) and stopping.")
                return best_dofs, residuals

            # Save best + optional early success
            if res_norm < best_res:
                best_res, best_step = res_norm, (fac + 1, it)
                best_dofs = jnp.array(applied_dofs)
                if best_res <= target_best:
                    print(f"[NH-early] Target reached: ||r||={best_res:.3e} "
                          f"at step {best_step[0]}, it {best_step[1]}. Stopping.")
                    return best_dofs, residuals

            # Divergence spike? → rollback & STOP
            if prev_res is not None and res_norm > growth_tol * prev_res:
                print(f"[NH-guard] Residual jump from {prev_res:.3e} to {res_norm:.3e} "
                      f"(>{growth_tol}×) at step {fac+1}, it {it}. "
                      f"Rolling back to best (||r||={best_res:.3e}) and stopping.")
                return best_dofs, residuals

            # Convergence for this substep?
            if res_norm < atol or (prev_res is not None and res_norm <= (1.0 - rtol) * prev_res):
                prev_res = res_norm
                if res_norm < atol:
                    break
            else:
                prev_res = res_norm

            # Newton update
            delta = fe_solver.LinearSolve(jac, res, applied_dofs)
            if not jnp.all(jnp.isfinite(delta)):
                print(f"[NH-guard] Non-finite Newton step at step {fac+1}, it {it}. "
                      f"Rolling back to best (||r||={best_res:.3e}) and stopping.")
                return best_dofs, residuals

            nd = fe_solver.fe_loss_function.non_dirichlet_indices
            applied_dofs = applied_dofs.at[nd].add(delta[nd])

    # Finished all substeps; return best we saw
    if best_res < np.inf:
        print(f"[NH-done] Finished all steps. Best ||r||={best_res:.3e} "
              f"at step {best_step[0]}, it {best_step[1]}.")
        return best_dofs, residuals
    else:
        print("[NH-done] No valid best state recorded; returning latest DOFs.")
        return applied_dofs, residuals
