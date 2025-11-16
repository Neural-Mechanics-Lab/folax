"""
 Authors: Kianoosh Taghikhani, https://github.com/Kianoosh1989
 Date: August, 2025
 License: FOL/LICENSE
"""
from  fol.controls.control import Control
import jax.numpy as jnp
from jax import jit,vmap
from functools import partial
from jax.nn import sigmoid
from fol.mesh_input_output.mesh import Mesh
from fol.loss_functions.loss import Loss
from fol.tools.decoration_functions import *
import jax
import numpy as np

class DirichletControl(Control):
    
    def __init__(self,control_name: str,control_settings: dict, fe_mesh: Mesh,fe_loss:Loss):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh
        self.loss_function = fe_loss
        

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        if self.initialized and not reinitialize:
            return

        self.dirichlet_values = self.loss_function.dirichlet_values
        self.dirichlet_indices = self.loss_function.dirichlet_indices
        dirichlet_indices_dict = self.loss_function.dirichlet_indices_dict

        learning_dirichlet_indices = []
        num_var = 0
        for dof in self.settings["learning_boundary"].keys():
            for learning_boundary_tag in self.settings["learning_boundary"][dof]:
                learning_dirichlet_indices.append(dirichlet_indices_dict[dof][learning_boundary_tag])
            num_var += 1
        self.learning_dirichlet_indices = np.array(learning_dirichlet_indices)

        self.dofs = self.loss_function.loss_settings.get("ordered_dofs")
        self.dirichlet_bc_dict = self.loss_function.loss_settings.get("dirichlet_bc_dict")
        self.dim = self.loss_function.loss_settings.get("compute_dims")
    
        self.num_control_vars = num_var  
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()
        self.initialized = True

    
    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self,variable_vector:jnp.array):
        dof_values = jnp.zeros(self.dim*self.fe_mesh.GetNumberOfNodes(), dtype=jnp.float32)
        dof_values = dof_values.at[self.dirichlet_indices].set(self.dirichlet_values)

        for i in range(self.num_control_vars):
            indices = np.ix_(self.learning_dirichlet_indices[i])
            dof_values = dof_values.at[indices].set(jnp.full(self.learning_dirichlet_indices[i].shape, variable_vector[i], dtype=jnp.float32))

        dirichlet_values = dof_values[self.dirichlet_indices]
        return dirichlet_values

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass
