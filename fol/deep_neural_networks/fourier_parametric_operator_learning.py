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

class FourierParametricOperatorLearning(DeepNetwork):

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
        batch_size = batch_X.shape[0]
        mesh_size = int(self.loss_function.fe_mesh.GetNumberOfNodes()**0.5)
        num_chs = int(batch_X.size/(mesh_size*mesh_size*batch_size))
        return nn_model(batch_X.reshape(batch_size,mesh_size,mesh_size,num_chs)).reshape(batch_size,-1)

    @print_with_timestamp_and_execution_time
    def Predict(self,batch_control:jnp.ndarray):
        control_outputs = self.control.ComputeBatchControlledVariables(batch_control)
        preds = self.ComputeBatchPredictions(control_outputs,self.flax_neural_network)
        return self.loss_function.GetFullDofVector(batch_control,preds)

    def Finalize(self):
        pass

class DataDrivenFourierParametricOperatorLearning(FourierParametricOperatorLearning):

    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
        batch_predictions = self.ComputeBatchPredictions(control_outputs,nn_model)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(batch[1],batch_predictions)
        loss_name = self.loss_function.GetName()
        return batch_loss, ({loss_name+"_min":batch_min,
                             loss_name+"_max":batch_max,
                             loss_name+"_avg":batch_avg,
                             "total_loss":batch_loss})
    
class PhysicsInformedFourierParametricOperatorLearning(FourierParametricOperatorLearning):

    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        control_outputs = self.control.ComputeBatchControlledVariables(batch[0])
        batch_predictions = self.ComputeBatchPredictions(control_outputs,nn_model)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(control_outputs,batch_predictions)
        loss_name = self.loss_function.GetName()
        return batch_loss, ({loss_name+"_min":batch_min,
                             loss_name+"_max":batch_max,
                             loss_name+"_avg":batch_avg,
                             "total_loss":batch_loss})