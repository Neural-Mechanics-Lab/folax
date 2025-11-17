"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: Feb, 2025
 License: FOL/LICENSE
"""

from typing import Iterator,Tuple 
import jax
import jax.numpy as jnp
from jax import jit,vmap
from functools import partial
from optax import GradientTransformation
from flax import nnx
from fol.deep_neural_networks.deep_network import DeepNetwork
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *
import orbax.checkpoint as ocp
import optax

class CategoricalDeepLearning(DeepNetwork):

    def __init__(self,
                 name:str,
                 flax_neural_network:nnx.Module,
                 optax_optimizer:GradientTransformation):

        self.name = name
        self.flax_neural_network = flax_neural_network
        self.optax_optimizer = optax_optimizer
        self.initialized = False
        
    @print_with_timestamp_and_execution_time
    def Initialize(self,reinitialize=False) -> None:

        if self.initialized and not reinitialize:
            return
        
        # create orbax checkpointer
        self.checkpointer = ocp.StandardCheckpointer()

        # initialize the nnx optimizer
        self.nnx_optimizer = nnx.Optimizer(self.flax_neural_network, self.optax_optimizer, wrt=nnx.Param)

        self.initialized = True
    
    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        return optax.softmax_cross_entropy_with_integer_labels(logits=nn_model(jnp.expand_dims(x_set[0], axis=0)).flatten(), labels=x_set[1][0])

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeBatchLossValue(self,batch_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        batch_losses = optax.softmax_cross_entropy_with_integer_labels(logits=nn_model(batch_set[0]),labels=batch_set[1].reshape(-1))
        total_mean_loss = jnp.mean(batch_losses)
        return total_mean_loss, ({"total_loss":total_mean_loss})

    @print_with_timestamp_and_execution_time
    @partial(jit, static_argnums=(0,))
    def Predict(self,batch_X):
        return self.flax_neural_network(batch_X)

    def Finalize(self):
        pass