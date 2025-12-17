"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: May, 2025
 License: FOL/LICENSE
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from flax import nnx
from fol.deep_neural_networks.meta_alpha_meta_implicit_parametric_operator_learning import MetaAlphaMetaImplicitParametricOperatorLearning

class DataDrivenMetaAlphaMetaImplicitParametricOperatorLearning(MetaAlphaMetaImplicitParametricOperatorLearning):

    def ComputeBatchPredictions(self,batch_X:jnp.ndarray,meta_model:Tuple[nnx.Module,nnx.Module]):

        nn_model,latent_step = meta_model

        latent_codes = jnp.zeros((batch_X.shape[0],nn_model.in_features))
        control_outputs = self.control.ComputeBatchControlledVariables(batch_X)

        def latent_loss(latent_code,control_output):
            nn_output = nn_model(latent_code[None, :],self.loss_function.fe_mesh.GetNodesCoordinates())
            return self.loss_function.ComputeBatchLoss(control_output,nn_output)[0]

        vec_grad_func = jax.vmap(jax.grad(latent_loss, argnums=0))
        for _ in range(self.num_latent_iterations):
            grads = vec_grad_func(latent_codes,control_outputs)
            latent_codes -= latent_step() * grads

        return nn_model(latent_codes,self.loss_function.fe_mesh.GetNodesCoordinates())
    
    def ComputeBatchLossValue(self,batch:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        batch_predictions = self.ComputeBatchPredictions(batch[0],nn_model)
        batch_loss,(batch_min,batch_max,batch_avg) = self.loss_function.ComputeBatchLoss(batch[1],batch_predictions)
        loss_name = self.loss_function.GetName()
        return batch_loss, ({loss_name+"_min":batch_min,
                             loss_name+"_max":batch_max,
                             loss_name+"_avg":batch_avg,
                             "total_loss":batch_loss})