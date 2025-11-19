"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2025
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

class DeepONetParametricOperatorLearning(DeepNetwork):

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
 
        if self.initialized and not reinitialize:
            return

        super().Initialize(reinitialize)

        if not self.control.initialized:
            self.control.Initialize(reinitialize)

        self.initialized = True

    def ComputeBatchPredictions(self,batch_X:jnp.ndarray,nn_model:nnx.Module):
        return nn_model(batch_X,self.loss_function.fe_mesh.GetNodesCoordinates()).reshape(batch_X.shape[0],-1)

    @print_with_timestamp_and_execution_time
    @partial(nnx.jit, static_argnums=(0,), donate_argnums=1)
    def Predict(self,batch_X):
        control_outputs = self.control.ComputeBatchControlledVariables(batch_X)
        preds = self.ComputeBatchPredictions(control_outputs,self.flax_neural_network)
        return self.loss_function.GetFullDofVector(batch_X,preds.reshape(preds.shape[0], -1))

    def Finalize(self):
        pass

class DataDrivenDeepONetParametricOperatorLearning(DeepONetParametricOperatorLearning):

    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
        batch_predictions = self.ComputeBatchPredictions(control_outputs,nn_model)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(batch[1],batch_predictions)
        loss_name = self.loss_function.GetName()
        return batch_loss, ({loss_name+"_min":batch_min,
                             loss_name+"_max":batch_max,
                             loss_name+"_avg":batch_avg,
                             "total_loss":batch_loss})
    
class PhysicsInformedDeepONetParametricOperatorLearning(DeepONetParametricOperatorLearning):

    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
        batch_predictions = self.ComputeBatchPredictions(control_outputs,nn_model)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(control_outputs,batch_predictions)
        loss_name = self.loss_function.GetName()
        return batch_loss, ({loss_name+"_min":batch_min,
                             loss_name+"_max":batch_max,
                             loss_name+"_avg":batch_avg,
                             "total_loss":batch_loss})
    
class PhysicsInformedDeepONetParametricOperatorLearningADLoss(DeepONetParametricOperatorLearning):

    def ComputeSingleModelSpatialGradient(self,nn_model:nnx.Module,nodes:jnp.array,control_output:jnp.array):
        coords = self.loss_function.fe_mesh.GetNodesCoordinates()
        def dnn_dx_i(i,control_output):
            return jax.jacfwd(nn_model,argnums=1)(control_output[None,:],coords[i,:].reshape(1,3)).squeeze()
        # loop over nodes
        return jax.vmap(dnn_dx_i, in_axes=(0, None))(nodes,control_output)
        
    def ComputeBatchModelSpatialGradient(self,nn_model:nnx.Module,nodes:jnp.array,control_outputs:jnp.array):    
        return jax.vmap(self.ComputeSingleModelSpatialGradient, in_axes=(None,None,0))(
                        nn_model,nodes,control_outputs) # shape: (batch_size, num_nodes, derivative_dim=3)
    
    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
        nn_derivative_batch = self.ComputeBatchModelSpatialGradient(nn_model,jnp.arange(self.loss_function.fe_mesh.GetNumberOfNodes()),control_outputs)
        batch_predictions = self.ComputeBatchPredictions(control_outputs,nn_model)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(nn_derivative_batch,control_outputs,batch_predictions)
        loss_name = self.loss_function.GetName()
        return batch_loss, ({loss_name+"_min":batch_min,
                             loss_name+"_max":batch_max,
                             loss_name+"_avg":batch_avg,
                             "total_loss":batch_loss})
