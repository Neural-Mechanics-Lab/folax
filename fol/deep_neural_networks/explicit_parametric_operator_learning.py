"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""

from typing import Iterator,Tuple 
import jax
import jax.numpy as jnp
from jax import jit,vmap
from functools import partial
from optax import GradientTransformation
from flax import nnx
from .deep_network import DeepNetwork
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *

class ExplicitParametricOperatorLearning(DeepNetwork):
    """
    A class for explicit parametric operator learning in deep neural networks.

    This class extends the `DeepNetwork` base class and is designed specifically 
    for learning parametric operators where spatial fields like predicted displacement
    are explicitly modeled. It inherits all the attributes and methods from `DeepNetwork` and introduces 
    additional components to handle control parameters.

    Attributes:
        name (str): The name assigned to the neural network model for identification purposes.
        control (Control): An instance of the Control class used for the parametric learning.
        loss_function (Loss): An instance of the Loss class representing the objective function to be minimized during training.
        flax_neural_network (Module): The Flax neural network model (inherited from flax.nnx.Module) that defines the architecture and forward pass of the network.
        optax_optimizer (GradientTransformation): The Optax optimizer used to compute and apply gradients during the training process.
        checkpoint_settings (dict): A dictionary of configurations used to manage checkpoints, saving model states and parameters during or after training. Defaults to an empty dictionary.
     
    """

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:nnx.Module,
                 optax_optimizer:GradientTransformation):
        super().__init__(name,loss_function,flax_neural_network,
                         optax_optimizer)
        self.control = control
        
    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize the explicit parametric operator learning model, its components, and control parameters.

        This method extends the initialization process defined in the `DeepNetwork` base class by
        ensuring that the control parameters used for parametric learning are also initialized.
        It handles both the initialization of core deep learning components (loss function, 
        checkpoint settings, neural network state restoration) and the initialization of 
        the control parameters essential for explicit parametric learning tasks.

        Parameters:
        ----------
        reinitialize : bool, optional
            If True, forces reinitialization of the model and its components, including control parameters,
            even if they have been initialized previously. Default is False.

        """

        if self.initialized and not reinitialize:
            return

        super().Initialize(reinitialize)

        if not self.control.initialized:
            self.control.Initialize(reinitialize)

        self.initialized = True

        # now check if the input output layers size match with 
        # loss and control sizes, this is explicit parametric learning
        if not hasattr(self.flax_neural_network, 'in_features'):
            fol_error(f"the provided flax neural netwrok does not have in_features "\
                      "which specifies the size of the input layer ") 

        if not hasattr(self.flax_neural_network, 'out_features'):
            fol_error(f"the provided flax neural netwrok does not have out_features "\
                      "which specifies the size of the output layer") 

        if self.flax_neural_network.in_features != self.control.GetNumberOfVariables():
            fol_error(f"the size of the input layer is {self.flax_neural_network.in_features} "\
                      f"does not match the size of control variables {self.control.GetNumberOfVariables()}")

        if self.flax_neural_network.out_features != self.loss_function.GetNumberOfUnknowns():
            fol_error(f"the size of the output layer is {self.flax_neural_network.out_features} " \
                      f" does not match the size of unknowns of the loss function {self.loss_function.GetNumberOfUnknowns()}")

    def ComputeBatchPredictions(self,batch_X:jnp.ndarray,nn_model:nnx.Module):
        return nn_model(batch_X)
    
    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
        batch_unknowns_predictions = self.ComputeBatchPredictions(batch[0],nn_model)
        batch_full_pred = jnp.zeros((batch[0].shape[0],self.loss_function.GetTotalNumberOfDOFs()))
        batch_full_pred = batch_full_pred.at[:,self.loss_function.non_dirichlet_indices].set(batch_unknowns_predictions)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(control_outputs,batch_full_pred)
        loss_name = self.loss_function.GetName()
        return batch_loss, ({loss_name+"_min":batch_min,
                             loss_name+"_max":batch_max,
                             loss_name+"_avg":batch_avg,
                             "total_loss":batch_loss})

    @print_with_timestamp_and_execution_time
    @partial(nnx.jit, static_argnums=(0,), donate_argnums=1)
    def Predict(self,batch_X):
        batch_unknowns_predictions = self.ComputeBatchPredictions(batch_X,self.flax_neural_network)
        batch_full_pred = jnp.zeros((batch_X.shape[0],self.loss_function.GetTotalNumberOfDOFs()))
        batch_full_pred = batch_full_pred.at[:,self.loss_function.non_dirichlet_indices].set(batch_unknowns_predictions)
        return self.loss_function.GetFullDofVector(batch_X,batch_full_pred.reshape(batch_full_pred.shape[0], -1))

    def Finalize(self):
        pass