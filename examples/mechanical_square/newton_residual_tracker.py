# fol/tools/newton_residual_tracker.py
import os
import numpy as np
import jax.numpy as jnp


class NewtonResidualTracker:
    """
    Tracks Newton residuals for a single FE solve and writes them to CSV.
    Residuals are always taken on the unknown (non-Dirichlet) DOFs.
    """

    def __init__(self, case_dir, sample_tag="baseline"):
        self.path = os.path.join(case_dir, f"newton_residuals_{sample_tag}.csv")
        with open(self.path, "w") as f:
            f.write("load_step,iter,res_l2,res_rms_unknown\n")

    def log(self, load_step, it, res_unknown_jax):
        """
        res_unknown_jax: residual vector restricted to non-Dirichlet DOFs.
        """
        res_unknown = np.asarray(res_unknown_jax)
        l2 = float(np.linalg.norm(res_unknown))
        rms = float(l2 / np.sqrt(res_unknown.size))

        print(
            f"[Newton] step {load_step:03d} it {it:02d} "
            f"||r||_2={l2:.3e}, rms={rms:.3e}"
        )

        with open(self.path, "a") as f:
            f.write(f"{load_step},{it},{l2:.8e},{rms:.8e}\n")

        return l2, rms


def custom_newton_solve(
    fe_solver,
    control_vars,
    initial_dofs,
    case_dir,
    sample_tag="baseline",
    target_best=1e-6,
    growth_tol=50.0,
):
    """
    Newton–Raphson with:
    - tracking of L2 and RMS residual on unknown DOFs
    - early-stop when 'target_best' is reached
    - guard against divergence spikes (growth_tol)
    Returns:
        best_dofs, residuals_rms, total_iters
    """
    fe_loss = fe_solver.fe_loss_function
    nd = fe_loss.non_dirichlet_indices

    tracker = NewtonResidualTracker(case_dir, sample_tag)

    residuals_rms = []
    applied_dofs = initial_dofs

    n_incr = fe_solver.nonlinear_solver_settings["load_incr"]
    maxit = fe_solver.nonlinear_solver_settings["maxiter"]
    atol = fe_solver.nonlinear_solver_settings["abs_tol"]
    rtol = fe_solver.nonlinear_solver_settings["rel_tol"]

    best_res = np.inf
    best_step = (0, 0)
    best_dofs = jnp.array(applied_dofs)
    total_iters = 0

    for fac in range(n_incr):
        load_factor = (fac + 1) / n_incr
        applied_dofs = fe_loss.ApplyDirichletBCOnDofVector(applied_dofs, load_factor)

        prev_res = None

        for it in range(1, maxit + 1):
            total_iters += 1

            # FE residual/Jacobian at current state
            jac, res = fe_loss.ComputeJacobianMatrixAndResidualVector(
                control_vars, applied_dofs
            )

            # Restrict to unknown DOFs
            res_unknown = res[nd]
            _, res_rms = tracker.log(
                load_step=fac + 1, it=it, res_unknown_jax=res_unknown
            )
            residuals_rms.append(res_rms)

            # Non-finite → rollback and stop
            if not np.isfinite(res_rms):
                print(
                    f"[NH-guard] non-finite residual at step {fac+1}, it {it}. "
                    f"Rollback to best (rms={best_res:.3e}) and stop."
                )
                return best_dofs, residuals_rms, total_iters

            # Update best-so-far
            if res_rms < best_res:
                best_res = res_rms
                best_step = (fac + 1, it)
                best_dofs = jnp.array(applied_dofs)
                if best_res <= target_best:
                    print(
                        f"[NH-early] target reached: rms={best_res:.3e} "
                        f"at step {best_step[0]}, it {best_step[1]}."
                    )
                    return best_dofs, residuals_rms, total_iters

            # Divergence spike → rollback and stop
            if prev_res is not None and res_rms > growth_tol * prev_res:
                print(
                    f"[NH-guard] residual jump from {prev_res:.3e} to {res_rms:.3e} "
                    f"(>{growth_tol}×) at step {fac+1}, it {it}. "
                    f"Rollback to best (rms={best_res:.3e}) and stop."
                )
                return best_dofs, residuals_rms, total_iters

            # Convergence / stagnation for this substep
            if res_rms < atol:
                break
            if prev_res is not None and res_rms >= (1.0 - rtol) * prev_res:
                # Relative improvement smaller than rtol → treat as stagnation
                break

            prev_res = res_rms

            # Newton update
            delta = fe_solver.LinearSolve(jac, res, applied_dofs)
            if not jnp.all(jnp.isfinite(delta)):
                print(
                    f"[NH-guard] non-finite Newton step at step {fac+1}, it {it}. "
                    f"Rollback to best (rms={best_res:.3e}) and stop."
                )
                return best_dofs, residuals_rms, total_iters

            applied_dofs = applied_dofs.at[nd].add(delta[nd])

    if best_res < np.inf:
        print(
            f"[NH-done] finished all steps. "
            f"Best rms={best_res:.3e} at step {best_step[0]}, it {best_step[1]}."
        )
        return best_dofs, residuals_rms, total_iters
    else:
        print("[NH-done] no valid best state recorded; returning latest DOFs.")
        return applied_dofs, residuals_rms, total_iters
