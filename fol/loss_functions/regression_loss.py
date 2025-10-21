"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: November, 2024
 License: FOL/LICENSE
"""
import jax.numpy as jnp
from fol.tools.decoration_functions import *
from fol.mesh_input_output.mesh import Mesh
from jax import jit,grad
from functools import partial
from  .loss import Loss

class RegressionLoss(Loss):
    """
    Regression loss class for managing loss computation in a finite element (FE) framework.

    This class is responsible for initializing and configuring the regression loss 
    with specific settings and a finite element mesh. It extends the base `Loss` 
    class and manages the degrees of freedom (DOFs) related to nodal unknowns in 
    the FE mesh.

    Attributes:
        loss_settings (dict): Dictionary containing settings for loss computation, 
            including nodal unknowns and other configuration parameters.
        fe_mesh (Mesh): Finite element mesh object used to define the spatial 
            structure for the regression loss.
        dofs: Degrees of freedom for the nodal unknowns as defined in `loss_settings`.

    Parameters:
        name (str): Name of the loss, used for identification.
        loss_settings (dict): Configuration dictionary for loss-related settings.
        fe_mesh (Mesh): Mesh object defining the finite element structure.
    """

    def __init__(self, name: str, loss_settings: dict, fe_mesh: Mesh) -> None:
        super().__init__(name)
        self.loss_settings = loss_settings
        self.fe_mesh = fe_mesh
        self.dofs = self.loss_settings["nodal_unknows"]

    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initializes the regression loss for the training process.

        This method prepares the loss by setting up necessary indices based on the degrees 
        of freedom (DOFs) and finite element mesh nodes. It is generally called only once 
        during the training process, but can be reinitialized if required by setting 
        `reinitialize=True`.

        Parameters:
            reinitialize (bool, optional): If True, allows reinitialization even if 
                the loss was already initialized. Default is False.
        """

        if self.initialized and not reinitialize:
            return
        
        self.non_dirichlet_indices = jnp.arange(len(self.dofs)*self.fe_mesh.GetNumberOfNodes())

        self.initialized = True

    def GetFullDofVector(self,known_dofs: jnp.array,unknown_dofs: jnp.array) -> jnp.array:
        return unknown_dofs

    def GetNumberOfUnknowns(self) -> int:
        pass

    # @print_with_timestamp_and_execution_time
    def ComputeBatchLoss(self,gt_values:jnp.array,pred_values:jnp.array) -> tuple[float, tuple[float, float, float]]:
        """
        Computes the regression loss between ground truth values and predicted values.

        This method calculates the mean squared error (MSE) between the ground truth (`gt_values`) 
        and predicted values (`pred_values`) and returns it as the primary loss measure. 
        Additionally, it provides a summary of the error range by returning the minimum, 
        maximum, and mean errors.

        Parameters:
            gt_values (jnp.array): Array of ground truth values.
            pred_values (jnp.array): Array of predicted values.

        Returns:
            tuple: A tuple containing:
                - mean error (float): The mean squared error between `gt_values` and `pred_values`.
                - error summary (tuple): A tuple with minimum error, maximum error, and mean error.

        """
        err = (gt_values-pred_values)**2
        return jnp.mean(err),(jnp.min(err),jnp.max(err),jnp.mean(err))

    def Finalize(self) -> None:
        """
        Finalizes the loss computation for the training process.

        This method performs any necessary cleanup or final adjustments to the loss 
        at the end of the training process. It is intended to be called only once 
        after all training iterations are completed.

        """

        pass



