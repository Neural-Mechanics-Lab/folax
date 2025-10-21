"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""
from abc import ABC, abstractmethod
import jax.numpy as jnp

class Loss(ABC):
    """Base abstract loss class.

    The base abstract control class has the following responsibilities.
        1. Initalizes and finalizes the loss.

    """
    def __init__(self, loss_name: str) -> None:
        self.__name = loss_name
        self.initialized = False

    def GetName(self) -> str:
        return self.__name

    @abstractmethod
    def Initialize(self) -> None:
        """Initializes the loss.

        This method initializes the loss. This is only called once in the whole training process.

        """
        pass

    @abstractmethod
    def GetFullDofVector(self,known_dofs: jnp.array,unknown_dofs: jnp.array) -> jnp.array:
        """Returns vector of all dofs.

        """
        pass

    @abstractmethod
    def GetNumberOfUnknowns(self) -> int:
        """Get number of unknowns, i.e., dofs

        """
        pass

    @abstractmethod
    def ComputeBatchLoss(self) -> None:
        """Computes the single loss.

        This method initializes the loss. This is only called once in the whole training process.

        """
        pass

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the loss.

        This method finalizes the loss. This is only called once in the whole training process.

        """
        pass



