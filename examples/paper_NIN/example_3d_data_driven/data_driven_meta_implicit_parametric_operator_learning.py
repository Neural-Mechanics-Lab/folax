"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2025
 License: FOL/LICENSE
"""

from typing import Tuple 
import jax
import jax.numpy as jnp
from functools import partial
from optax import GradientTransformation
from flax import nnx
from fol.deep_neural_networks.meta_implicit_parametric_operator_learning import MetaImplicitParametricOperatorLearning
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *
from fol.deep_neural_networks.nns import HyperNetwork

class DataDrivenMetaImplicitParametricOperatorLearning(MetaImplicitParametricOperatorLearning):
  
    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,orig_features:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
     
        latent_code = jnp.zeros(nn_model.in_features)

        @jax.jit
        def loss(input_latent_code):
            nn_output = nn_model(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
            return self.loss_function.ComputeSingleLoss(orig_features[0],nn_output)[0]

        loss_latent_grad_fn = jax.grad(loss)
        for _ in range(self.num_latent_iterations):
            grads = loss_latent_grad_fn(latent_code)
            latent_code -= self.latent_step * grads / jnp.linalg.norm(grads)
        
        nn_output = nn_model(latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        return self.loss_function.ComputeSingleLoss(orig_features[1],nn_output)

