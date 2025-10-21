"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""
from abc import ABC, abstractmethod
import jax.numpy as jnp
from functools import partial
from jax import jit,jacfwd
import jax
from fol.tools.decoration_functions import *

class Control(ABC):
    """Base abstract control class.

    The base abstract control class has the following responsibilities.
        1. Initalizes and finalizes the model.

    """
    def __init__(self, control_name: str) -> None:
        self.__name = control_name
        self.initialized = False
        self.num_control_vars = None
        self.num_controlled_vars = None

    def GetName(self) -> str:
        return self.__name

    @abstractmethod
    def Initialize(self) -> None:
        """Initializes the control.

        This method initializes the control. This is only called once in the whole training process.

        """
        pass

    def GetNumberOfVariables(self):
        return self.num_control_vars
    
    def GetNumberOfControlledVariables(self):
        return self.num_controlled_vars

    @abstractmethod
    def ComputeControlledVariables(self,variable_vector:jnp.array) -> None:
        """Computes the controlled variables for the given variables.

        """
        pass

    def ComputeBatchControlledVariables(self,batch_variable_vector:jnp.array) -> None:
        """Computes the controlled variables for the given batch variables.

        """
        return jnp.squeeze(jax.vmap(self.ComputeControlledVariables,(0))(batch_variable_vector))

    @partial(jit, static_argnums=(0,))
    def ComputeJacobian(self,control_vec):
        return jnp.squeeze(jacfwd(self.ComputeControlledVariables,argnums=0)(control_vec))

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the control.

        This method finalizes the control. This is only called once in the whole training process.

        """
        pass



