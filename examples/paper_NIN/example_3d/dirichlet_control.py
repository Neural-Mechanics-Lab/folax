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

class DirichletControl3D(Control):
    
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
        dof_values = jnp.zeros(3*self.fe_mesh.GetNumberOfNodes(), dtype=jnp.float64)
        dof_values = dof_values.at[self.dirichlet_indices].set(self.dirichlet_values)

        for i in range(self.num_control_vars):
            indices = np.ix_(self.learning_dirichlet_indices[i])
            dof_values = dof_values.at[indices].set(jnp.full(self.learning_dirichlet_indices[i].shape, variable_vector[i], dtype=jnp.float64))

        dirichlet_values = dof_values[self.dirichlet_indices]
        return dirichlet_values

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass

class DirichletControl2D(Control):
    
    def __init__(self,control_name: str,control_settings: dict, fe_mesh: Mesh,fe_loss:Loss):
        super().__init__(control_name)
        self.settings = control_settings
        self.fe_mesh = fe_mesh
        self.loss_function = fe_loss
        

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        if self.initialized and not reinitialize:
            return

        self.boundary_node_ids = jnp.array(self.fe_mesh.GetNodeSet("right"))
        self.dirichlet_indices = self.loss_function.dirichlet_indices
        dirichlet_indices_dict = self.loss_function.dirichlet_indices_dict
        self.right_indices_ux = dirichlet_indices_dict["Ux"]['right']
        self.right_indices_uy = dirichlet_indices_dict["Uy"]['right']
       
        self.dofs = self.loss_function.loss_settings.get("ordered_dofs")
        self.dirichlet_bc_dict = self.loss_function.loss_settings.get("dirichlet_bc_dict")
        self.dim = self.loss_function.loss_settings.get("compute_dims")
    
        self.num_control_vars = 2   #len(self.dirichlet_indices) / 2
        self.num_controlled_vars = self.fe_mesh.GetNumberOfNodes()
        self.initialized = True

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self,variable_vector:jnp.array):
        dof_values = jnp.zeros(2*self.fe_mesh.GetNumberOfNodes(), dtype=jnp.float32)
        dof_values = dof_values.at[self.dirichlet_indices].set(self.dirichlet_values)
        dof_values = dof_values.at[self.right_indices_ux].set(jnp.full(self.boundary_node_ids.shape, variable_vector[0], dtype=jnp.float32))
        dof_values = dof_values.at[self.right_indices_uy].set(jnp.full(self.boundary_node_ids.shape, variable_vector[1], dtype=jnp.float32))

        dirichlet_values = dof_values[self.dirichlet_indices]
        return dirichlet_values

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass
