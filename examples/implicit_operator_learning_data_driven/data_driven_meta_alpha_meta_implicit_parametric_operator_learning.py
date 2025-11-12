"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2025
 License: FOL/LICENSE
"""

from typing import Tuple,Iterator
import jax
import jax.numpy as jnp
import optax
from functools import partial
from optax import GradientTransformation
from flax import nnx
from tqdm import trange
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *
from fol.deep_neural_networks.nns import HyperNetwork

class LatentStepModel(nnx.Module):
    def __init__(self, init_latent_step_value):
        self.latent_step = nnx.Param(init_latent_step_value)
    def __call__(self):
        return self.latent_step 

class DataDrivenMetaAlphaMetaImplicitParametricOperatorLearning(MetaAlphaMetaImplicitParametricOperatorLearning):

    @partial(nnx.jit, static_argnums=(0,))
    def TrainStep(self, meta_state, data):
        nn_model, main_optimizer, latent_step_model, latent_optimizer = meta_state
        meta_model = (nn_model,latent_step_model)

        @nnx.jit
        def compute_batch_loss(batch_XY,meta_model):
            nn_model, latent_step_model = meta_model
            latent_codes = self.ComputeBatchLatent(batch_XY[0],nn_model,latent_step_model)
            return self.ComputeBatchLossValue((latent_codes,batch_XY[1]),nn_model)[0],latent_codes
        
        (loss_value,latent_codes),meta_grads = nnx.value_and_grad(compute_batch_loss,argnums=1,has_aux=True) (data,meta_model)
        main_optimizer.update(nn_model,meta_grads[0])
        latent_optimizer.update(latent_step_model,meta_grads[1])
        return loss_value
    
    @partial(nnx.jit, static_argnums=(0,))
    def TestStep(self, meta_state, data):
        nn_model, main_optimizer, latent_step_model, latent_optimizer = meta_state
        latent_codes = self.ComputeBatchLatent(data[0],nn_model,latent_step_model)
        return self.ComputeBatchLossValue((latent_codes,data[1]),nn_model)[0]

    @print_with_timestamp_and_execution_time
    def Predict(self,batch_control:jnp.ndarray):
        batch_X = jax.vmap(self.control.ComputeControlledVariables)(batch_control)
        latent_codes = self.ComputeBatchLatent(batch_control,self.flax_neural_network,self.latent_step_nnx_model)
        batch_Y =jax.vmap(self.flax_neural_network,(0,None))(latent_codes,self.loss_function.fe_mesh.GetNodesCoordinates())
        batch_Y = batch_Y.reshape(latent_codes.shape[0], -1)[:,self.loss_function.non_dirichlet_indices]
        return jax.vmap(self.loss_function.GetFullDofVector)(batch_X,batch_Y)
