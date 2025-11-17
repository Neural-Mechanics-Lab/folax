"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: July, 2024
 License: FOL/LICENSE
"""
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
from  .fe_linear_residual_based_solver import FiniteElementLinearResidualBasedSolver
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *
from fol.loss_functions.fe_loss import FiniteElementLoss

class FiniteElementNonLinearResidualBasedSolver(FiniteElementLinearResidualBasedSolver):
    """Nonlinear solver class.

    """

    @print_with_timestamp_and_execution_time
    def __init__(self, fe_solver_name: str, fe_loss_function: FiniteElementLoss, fe_solver_settings:dict={}) -> None:
        super().__init__(fe_solver_name,fe_loss_function,fe_solver_settings)
        self.nonlinear_solver_settings = {"rel_tol":1e-8,
                                           "abs_tol":1e-8,
                                           "maxiter":20,
                                           "load_incr":5}

    @print_with_timestamp_and_execution_time
    def Initialize(self) -> None:
        super().Initialize() 
        if "nonlinear_solver_settings" in self.fe_solver_settings.keys():
            self.nonlinear_solver_settings = UpdateDefaultDict(self.nonlinear_solver_settings,
                                                                self.fe_solver_settings["nonlinear_solver_settings"])
    @print_with_timestamp_and_execution_time
    def Solve(self,current_control_vars,current_dofs_np:np.array):
        current_dofs = jnp.array(current_dofs_np)
        load_increament = self.nonlinear_solver_settings["load_incr"]
        write_resnorm = []
        write_p11 = []
        for load_fac in range(load_increament):
            fol_info(f"loadStep; increment:{load_fac+1}")
            applied_BC_dofs = self.fe_loss_function.ApplyDirichletBCOnDofVector(current_dofs,(load_fac+1)/load_increament)
            for i in range(self.nonlinear_solver_settings["maxiter"]):
                BC_applied_jac,BC_applied_r = self.fe_loss_function.ComputeJacobianMatrixAndResidualVector(
                                                                    current_control_vars,applied_BC_dofs)
                res_norm = jnp.linalg.norm(BC_applied_r,ord=2)
                if jnp.isnan(res_norm):
                    fol_info("Residual norm is NaN, check inputs!")
                    raise(ValueError("res_norm contains nan values!"))
                    
                delta_dofs = self.LinearSolve(BC_applied_jac,BC_applied_r,applied_BC_dofs)
                delta_norm = jnp.linalg.norm(delta_dofs,ord=2)
                applied_BC_dofs = applied_BC_dofs.at[:].add(delta_dofs)

                if self.fe_solver_settings.get('output_directory') is not None:
                    write_resnorm.append(res_norm)
                    np.savetxt(os.path.join(self.fe_solver_settings['output_directory'],'res_norm_jax.txt'),np.array(write_resnorm))
                
                if delta_norm<self.nonlinear_solver_settings["rel_tol"] or delta_norm<self.nonlinear_solver_settings["abs_tol"]:
                    fol_info(f"converged; iterations:{i+1},delta_norm:{delta_norm},residuals_norm:{res_norm}")
                    break
                if res_norm<self.nonlinear_solver_settings["rel_tol"] or res_norm<self.nonlinear_solver_settings["abs_tol"]:
                    fol_info(f"converged; iterations:{i+1},delta_norm:{delta_norm},residuals_norm:{res_norm}")
                    break
                elif i+1==self.nonlinear_solver_settings["maxiter"]:
                    fol_info(f"maximum num iterations:{i+1} acheived,delta_norm:{delta_norm},residuals_norm:{res_norm}")
                    break
                else:
                    fol_info(f"iteration:{i+1},delta_norm:{delta_norm},residuals_norm:{res_norm}")
                
            current_dofs = current_dofs.at[self.fe_loss_function.non_dirichlet_indices].set(applied_BC_dofs[self.fe_loss_function.non_dirichlet_indices])
            current_dofs = current_dofs.at[self.fe_loss_function.dirichlet_indices].set(applied_BC_dofs[self.fe_loss_function.dirichlet_indices])
        return applied_BC_dofs






