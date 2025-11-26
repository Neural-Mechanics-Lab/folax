"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: July, 2024
 License: FOL/LICENSE
"""
import jax.numpy as jnp
from  .fe_nonlinear_residual_based_solver import FiniteElementNonLinearResidualBasedSolver
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *


class FiniteElementNonLinearResidualBasedSolverWithStateUpdate(FiniteElementNonLinearResidualBasedSolver):
    """
    Nonlinear finite element solver with incremental Newton–Raphson iterations
    and Gauss-point state variable updates.

    This solver extends `FiniteElementNonLinearResidualBasedSolver` by performing
    a coupled global–local update:
    
    • Global Newton–Raphson iterations solve for global DOFs.
    • Local Gauss-point state variables (e.g., internal variables, history fields)
      are updated each iteration and committed upon convergence.
    • Load is applied incrementally according to `nonlinear_solver_settings["load_incr"]`.
    • Convergence is checked using absolute residual tolerance, relative update 
      tolerance, and maximum Newton iterations.
    • Convergence history (residual norms, ΔDOF norms) is recorded for every load step.
    • Returns either the final load step only or the full history over all steps.

    This class is suitable for nonlinear problems requiring history-dependent 
    state updates (e.g., elastoplasticity, damage, viscoelasticity, etc.).

    """
    @print_with_timestamp_and_execution_time
    def Solve(self,current_control_vars:jnp.array,current_dofs:jnp.array,current_state:jnp.array=None,return_all_steps:bool=False):
        """
        Solve the nonlinear finite element system using an incremental,
        residual-based Newton–Raphson procedure with internal state updates.

        The algorithm applies load in increments, performs Newton iterations
        for each load step, updates Gauss-point state variables, and records
        convergence metrics.

        Parameters
        ----------
        current_control_vars : jax.numpy.ndarray
            Array of control variables (e.g., material parameters like heterogeneity).
        
        current_dofs : jax.numpy.ndarray
            Initial solution vector (or DOF vector). Shape: (n_dofs,).

        current_state : jax.numpy.ndarray, optional
            Initial Gauss-point state variables of shape:
                (n_elements, n_gauss_points, n_state_vars)
            If None, a zero state field is initialized.

        return_all_steps : bool, optional (default=False)
            Controls what is returned:
            • If False : returns only the final load step DOFs and state.
            • If True  : returns a stacked array of solutions and states for
                         every load step.

        Notes
        -----
        • Convergence is detected when:
              - residual norm < abs_tol
              - update norm   < rel_tol
              - iteration count reaches maxiter
        • Severe mesh distortion or invalid material response may result in
          a NaN residual; the solver detects this and raises an informative
          error.
        • Uses `LinearSolve()` to solve the Newton system.
        • Dirichlet boundary conditions are applied incrementally per load step.

        """
        current_dofs = jnp.asarray(current_dofs)
        current_control_vars = jnp.asarray(current_control_vars)
        num_load_steps = self.nonlinear_solver_settings["load_incr"]
        # --- init per-GP state (committed) once ---
        nelem = self.fe_loss_function.fe_mesh.GetNumberOfElements(self.fe_loss_function.element_type)
        ngp   = self.fe_loss_function.fe_element.GetIntegrationData()[0].shape[0]
        if current_state is None:
            current_state = jnp.zeros((nelem, ngp, 4)) 
        else:
            current_state = jnp.asarray(current_state)

        solution_history_dict = {}
        for load_step in range(1,num_load_steps+1):
            load_step_value = (load_step)/num_load_steps
            # increment load
            current_dofs = self.fe_loss_function.ApplyDirichletBCOnDofVector(current_dofs,load_step_value)
            newton_converged = False
            solution_history_dict[load_step] = {"res_norm":[],"delta_dofs_norm":[]}
            for i in range(1,self.nonlinear_solver_settings["maxiter"]+1):
                new_state,BC_applied_jac,BC_applied_r = self.fe_loss_function.ComputeJacobianMatrixAndResidualVector(
                                                                    current_control_vars,current_dofs,old_state_gps=current_state)
                
                # check residuals norm
                res_norm = jnp.linalg.norm(BC_applied_r,ord=2)
                if jnp.isnan(res_norm):
                    fol_info(
                        "\n"
                        "──────────────────── NEWTON ERROR ────────────────────\n"
                        "  Residual norm has become NaN.\n"
                        "  Possible causes:\n"
                        "    • Divergent Newton iteration\n"
                        "    • Inconsistent or ill-posed boundary conditions\n"
                        "    • Invalid / non-physical material parameters or state\n"
                        "    • Singular or severely ill-conditioned stiffness matrix\n"
                        "    • Severely distorted mesh (element quality breakdown)\n"
                        "───────────────────────────────────────────────────────\n"
                    )
                    raise ValueError("Residual norm contains NaN values.")

                # linear solve and calculate nomrs 
                delta_dofs = self.LinearSolve(BC_applied_jac,BC_applied_r,current_dofs)
                delta_norm = jnp.linalg.norm(delta_dofs,ord=2)

                newton_converged = (
                    res_norm < self.nonlinear_solver_settings["abs_tol"] or
                    delta_norm < self.nonlinear_solver_settings["rel_tol"] or
                    i == self.nonlinear_solver_settings["maxiter"]
                )

                indent = " " * 5
                fol_info(
                    f"\n"
                    f"{indent} ───────── Load Step {load_step} ─────────\n"
                    f"{indent}   Newton Iteration : {i} (max = {self.nonlinear_solver_settings["maxiter"]})\n"
                    f"{indent}   Residual Norm    : {res_norm:.3e} (abs_tol = {self.nonlinear_solver_settings["abs_tol"]:.3e})\n"
                    f"{indent}   Δ DOFs Norm      : {delta_norm:.3e} (rel_tol = {self.nonlinear_solver_settings["rel_tol"]:.3e})\n"
                    f"{indent}   Converged        : {'True' if newton_converged else 'False'}\n"
                    f"{indent}────────────────────────────────────────────"
                )

                solution_history_dict[load_step]["res_norm"].append(res_norm)
                solution_history_dict[load_step]["delta_dofs_norm"].append(delta_norm)
                
                if newton_converged:
                    break
                
                # if not converged update
                current_dofs = current_dofs.at[:].add(delta_dofs)
                current_state = new_state

            if return_all_steps:
                # Initialize on first load step
                if load_step == 1:
                    load_steps_solutions = jnp.copy(current_dofs)
                    load_steps_states = jnp.copy(current_state[None, ...])
                else:
                    load_steps_solutions = jnp.vstack([load_steps_solutions, current_dofs])
                    load_steps_states = jnp.vstack([load_steps_states, current_state[None, ...]])
            else:
                # Always only return the last step
                load_steps_solutions = jnp.copy(current_dofs)
                load_steps_states = jnp.copy(current_state)
            
            self.PlotHistoryDict(solution_history_dict)

        return load_steps_solutions, load_steps_states, solution_history_dict

