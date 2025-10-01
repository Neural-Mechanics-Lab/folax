"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: October, 2024
 License: FOL/LICENSE
"""
from  .control import Control
import jax.numpy as jnp
from jax import jit
from functools import partial
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh

class IdentityControl(Control):
    
    def __init__(self,control_name: str,control_settings: dict, num_vars):
        super().__init__(control_name)
        self.num_vars = num_vars
        self.settings = control_settings

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        if self.initialized and not reinitialize:
            self.initialized = True
        self.num_control_vars = self.num_vars
        self.num_controlled_vars = self.num_vars

    @partial(jit, static_argnums=(0,))
    def ComputeControlledVariables(self,variable_vector:jnp.array):
        return variable_vector

    @print_with_timestamp_and_execution_time
    def Finalize(self) -> None:
        pass