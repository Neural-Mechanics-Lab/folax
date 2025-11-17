"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: October, 2024
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

class ImplicitParametricOperatorLearning(DeepNetwork):
    """
    A class for implicit parametric operator learning in deep neural networks.

    This class extends the `DeepNetwork` base class and is designed specifically 
    for learning implicit parametric operators. These operators account for control 
    parameters that influence the spatial fields, such as predicted displacements or 
    other modeled phenomena. The class inherits all attributes and methods from 
    `DeepNetwork` and introduces additional components for handling parametric inputs.

    Attributes:
        name (str): Identifier for the neural network model.
        control (Control): An instance of the `Control` class for managing parametric inputs 
            and influencing model predictions.
        loss_function (Loss): An instance of the `Loss` class defining the objective 
            function for training.
        flax_neural_network (nnx.Module): The Flax-based neural network module that specifies 
            the architecture and forward computation.
        optax_optimizer (GradientTransformation): The Optax optimizer used to manage 
            gradient updates during training.
        checkpoint_settings (dict): Configuration dictionary for checkpointing, specifying 
            how and when to save model states and parameters during training. Defaults to an empty dictionary.
        working_directory (str): Directory path where model files, checkpoints, and logs 
            will be stored. Defaults to the current directory ('.').
    """

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:nnx.Module,
                 optax_optimizer:GradientTransformation):
        """
        Initializes an instance of the `ImplicitParametricOperatorLearning` class.

        Parameters:
            name (str): The name assigned to the neural network model.
            control (Control): The control mechanism or parameters guiding 
                parametric operator learning.
            loss_function (Loss): The objective function to be minimized during training.
            flax_neural_network (nnx.Module): The Flax-based neural network module defining 
                the model's architecture and behavior.
            optax_optimizer (GradientTransformation): The optimizer for updating model weights 
                based on gradient information.
            checkpoint_settings (dict, optional): Configuration for managing model checkpoints. 
                Defaults to an empty dictionary.
            working_directory (str, optional): Path to the directory where files will be stored. 
                Defaults to the current directory ('.').
        """
        super().__init__(name,loss_function,flax_neural_network,
                         optax_optimizer)
        self.control = control
        
    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize the implicit parametric operator learning model and its components.

        This method extends the initialization process defined in the `DeepNetwork` base class 
        by including the initialization of control parameters critical for parametric learning. 
        It ensures that all components, such as the loss function, checkpoint settings, 
        neural network state, and control parameters, are correctly set up. The method also validates 
        the consistency between the neural network architecture and the dimensions of the loss 
        function and control parameters.

        Parameters:
        ----------
        reinitialize : bool, optional
            If True, forces the reinitialization of all components, including the control parameters, 
            even if they have already been initialized. Default is False.

        Raises:
        -------
        ValueError:
            If the neural network's input (`in_features`) or output (`out_features`) dimensions 
            do not match the expected sizes based on the control parameters and loss function.
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

        if self.flax_neural_network.out_features != len(self.loss_function.dofs):
            fol_error(f"the size of the output layer is {self.flax_neural_network.out_features} " \
                      f" does not match the number of the loss function {self.loss_function.dofs}")

        # if self.flax_neural_network.in_features != self.control.GetNumberOfVariables():
        #     fol_error(f"the size of the input layer is {self.flax_neural_network.in_features} "\
        #               f"does not match the input size implicit/neural field which is {self.control.GetNumberOfVariables() + 3}")

    def ComputeBatchPredictions(self,batch_X:jnp.ndarray,nn_model:nnx.Module):
        return nn_model(batch_X,self.loss_function.fe_mesh.GetNodesCoordinates())

    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
        batch_predictions = self.ComputeBatchPredictions(batch[0],nn_model)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(control_outputs,batch_predictions)
        loss_name = self.loss_function.GetName()
        return batch_loss, ({loss_name+"_min":batch_min,
                             loss_name+"_max":batch_max,
                             loss_name+"_avg":batch_avg,
                             "total_loss":batch_loss})

    @print_with_timestamp_and_execution_time
    @partial(nnx.jit, static_argnums=(0,), donate_argnums=1)
    def Predict(self,batch_X):
        control_outputs = self.control.ComputeBatchControlledVariables(batch_X)
        preds = self.ComputeBatchPredictions(control_outputs,self.flax_neural_network)
        return self.loss_function.GetFullDofVector(control_outputs,preds.reshape(preds.shape[0], -1))

    @print_with_timestamp_and_execution_time
    @partial(nnx.jit, donate_argnums=(1,), static_argnums=(0,2))
    def PredictDynamics(self,initial_Batch:jnp.ndarray,num_steps:int):

        def step_fn(current_state, _):
            """Compute the next state given the current state."""
            next_state = self.Predict(current_state)
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_Batch, None, length=num_steps)

        # Stack the initial state with the predicted trajectory
        return jnp.vstack([jnp.expand_dims(initial_Batch, axis=0), trajectory])

    def Finalize(self):
        pass