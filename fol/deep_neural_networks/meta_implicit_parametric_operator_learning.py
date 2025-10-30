"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: December, 2024
 License: FOL/LICENSE
"""

from typing import Tuple 
import jax
import jax.numpy as jnp
from functools import partial
from optax import GradientTransformation
from flax import nnx
from .implicit_parametric_operator_learning import ImplicitParametricOperatorLearning
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *
from .nns import HyperNetwork

class MetaImplicitParametricOperatorLearning(ImplicitParametricOperatorLearning):
    """
    A meta-learning framework for implicit parametric operator learning.

    This class extends `ImplicitParametricOperatorLearning` by introducing meta-learning capabilities, 
    such as latent loop optimization. It enables efficient learning of parametric operators using deep 
    neural networks and advanced optimization techniques.

    Attributes
    ----------
    name : str
        The name of the neural network model for identification purposes.
    control : Control
        An instance of the `Control` class that manages the parametric learning process.
    loss_function : Loss
        The objective function to minimize during training.
    flax_neural_network : HyperNetwork
        The Flax-based hypernetwork model defining the architecture and forward pass.
    main_loop_optax_optimizer : GradientTransformation
        The Optax optimizer used for the primary optimization loop.
    latent_step : float
        Step size for latent loop optimization.
    num_latent_iterations : int
        Number of iterations for latent loop optimization.
    checkpoint_settings : dict
        Configuration dictionary for managing checkpoints. Defaults to an empty dictionary.
    working_directory : str
        Path to the working directory where model outputs and checkpoints are saved. Defaults to the current directory.
    """

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:HyperNetwork,
                 main_loop_optax_optimizer:GradientTransformation,
                 latent_step_size:float=1e-2,
                 num_latent_iterations:int=3):
        """
        Initializes the `MetaImplicitParametricOperatorLearning` class.

        This constructor sets up the meta-learning framework by initializing attributes and configurations 
        needed for training and optimization, including latent loop parameters.

        Parameters
        ----------
        name : str
            The name assigned to the neural network model for identification purposes.
        control : Control
            An instance of the `Control` class that manages the parametric learning process.
        loss_function : Loss
            An instance of the `Loss` class representing the objective function to minimize.
        flax_neural_network : HyperNetwork
            The Flax-based hypernetwork model defining the architecture and forward pass.
        main_loop_optax_optimizer : GradientTransformation
            The Optax optimizer for the primary optimization loop.
        latent_step_size : float, optional
            The step size for latent loop optimization. Default is 1e-2.
        num_latent_iterations : int, optional
            The number of iterations for latent loop optimization. Default is 3.

        Returns
        -------
        None
        """
        super().__init__(name,control,loss_function,flax_neural_network,
                         main_loop_optax_optimizer)
        
        self.latent_step = latent_step_size
        self.num_latent_iterations = num_latent_iterations

    def ComputeBatchPredictions(self,batch_X:jnp.ndarray,nn_model:nnx.Module):
        latent_codes = jnp.zeros((batch_X.shape[0],nn_model.in_features))
        control_outputs = self.control.ComputeBatchControlledVariables(batch_X)

        def latent_loss(latent_code,control_output):
            nn_output = nn_model(latent_code[None, :],self.loss_function.fe_mesh.GetNodesCoordinates())
            return self.loss_function.ComputeBatchLoss(control_output,nn_output)[0]

        vec_grad_func = jax.vmap(jax.grad(latent_loss, argnums=0))
        for _ in range(self.num_latent_iterations):
            grads = vec_grad_func(latent_codes,control_outputs)
            grads_norms =  jnp.linalg.norm(grads, axis=1, keepdims=True)
            norm_grads = grads/grads_norms
            latent_codes -= self.latent_step * norm_grads

        return nn_model(latent_codes,self.loss_function.fe_mesh.GetNodesCoordinates())

