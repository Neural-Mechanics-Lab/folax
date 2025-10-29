"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: December, 2024
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
from .implicit_parametric_operator_learning import ImplicitParametricOperatorLearning
from fol.tools.decoration_functions import *
from fol.loss_functions.loss import Loss
from fol.controls.control import Control
from fol.tools.usefull_functions import *
from .nns import HyperNetwork

class LatentStepModel(nnx.Module):
    def __init__(self, init_latent_step_value):
        self.latent_step = nnx.Param(init_latent_step_value)
    def __call__(self):
        return self.latent_step 

class MetaAlphaMetaImplicitParametricOperatorLearning(ImplicitParametricOperatorLearning):
    """
    A class for implementing meta-learning techniques in the context of implicit parametric operator learning.

    This class extends the `ImplicitParametricOperatorLearning` class and incorporates 
    meta-learning functionality for optimizing latent variables. It supports custom loss functions, 
    neural network models, and optimizers. Additionally, this class optimizes both the latent code 
    and the latent step size during the process of latent finding and optimization.

    Attributes:
        name (str): Name of the learning instance.
        control (Control): Control object to manage configurations and settings.
        loss_function (Loss): Loss function used for optimization.
        flax_neural_network (HyperNetwork): Neural network model for operator learning.
        main_loop_optax_optimizer (GradientTransformation): Optimizer for the main training loop.
        latent_step_optax_optimizer (GradientTransformation): Optimizer for updating latent variables.
        latent_step (float): Step size for latent updates.
        num_latent_iterations (int): Number of iterations for latent variable optimization.
        checkpoint_settings (dict): Settings for checkpointing, such as saving and restoring states.
        working_directory (str): Directory for saving files and logs.
        latent_step_optimizer_state: Internal state of the latent step optimizer.
        default_checkpoint_settings (dict): Default checkpoint settings, including directories and restore options.
    """

    def __init__(self,
                 name:str,
                 control:Control,
                 loss_function:Loss,
                 flax_neural_network:HyperNetwork,
                 main_loop_optax_optimizer:GradientTransformation,
                 latent_step_optax_optimizer:GradientTransformation,
                 latent_step_size:float=1e-2,
                 num_latent_iterations:int=3
                 ):
        """
        Initializes the MetaAlphaMetaImplicitParametricOperatorLearning instance.

        Args:
            name (str): Name of the learning instance.
            control (Control): Control object to manage configurations and settings.
            loss_function (Loss): Loss function used for optimization.
            flax_neural_network (HyperNetwork): Neural network model for operator learning.
            main_loop_optax_optimizer (GradientTransformation): Optimizer for the main training loop.
            latent_step_optax_optimizer (GradientTransformation): Optimizer for updating latent variables and step size.
            latent_step_size (float, optional): Initial step size for latent updates. Default is 1e-2.
            num_latent_iterations (int, optional): Number of iterations for latent variable optimization. Default is 3.
            checkpoint_settings (dict, optional): Settings for checkpointing, such as saving and restoring states. 
                                                  Default is an empty dictionary.
            working_directory (str, optional): Directory for saving files and logs. Default is '.'.

        Notes:
            This class not only finds the optimal latent code but also optimizes the latent step size 
            during the process of latent finding and optimization. This dual optimization ensures better 
            convergence and adaptability for varying problem conditions.
        """
        super().__init__(name,control,loss_function,flax_neural_network,
                         main_loop_optax_optimizer)
        
        self.latent_step_optimizer = latent_step_optax_optimizer
        self.latent_step_nnx_model = LatentStepModel(latent_step_size)
        self.num_latent_iterations = num_latent_iterations
        self.latent_nnx_optimizer = nnx.Optimizer(self.latent_step_nnx_model,self.latent_step_optimizer)

    def Finalize(self):
        pass
    
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

    def ComputeBatchLossValue(self,batch_data:Tuple[jnp.ndarray,jnp.ndarray],meta_model:nnx.Module):
        control_outputs = self.control.ComputeBatchControlledVariables(batch_data[0])
        return self.loss_function.ComputeBatchLoss(control_outputs,self.ComputeBatchPredictions(batch_data[0],meta_model))[0]

    def SaveCheckPoint(self,check_point_type,checkpoint_state_dir):
        """
        Saves the current state of the neural network to a specified directory.

        This method stores the state of the neural network model in a designated directory, ensuring the model's 
        state can be restored later.

        Parameters
        ----------
        None

        Returns
        -------
        None
            The current state of the neural network is saved to the specified directory. A confirmation message is 
            logged to indicate the successful save operation.

        Notes
        -----
        - The directory for saving the checkpoint is specified in the `checkpoint_settings` attribute under the 
        `state_directory` key.
        - The directory path is converted to an absolute path before saving.
        - Uses the `self.checkpointer.save` method to store the state and forces the save operation.
        - Logs the save operation using `fol_info`.
        """
        
        nn_checkpoint_state_dir = checkpoint_state_dir + "/nn"
        absolute_path = os.path.abspath(nn_checkpoint_state_dir)
        self.checkpointer.save(absolute_path, nnx.state(self.flax_neural_network),force=True)

        latent_checkpoint_state_dir = checkpoint_state_dir + "/latent"
        absolute_path = os.path.abspath(latent_checkpoint_state_dir)
        self.checkpointer.save(absolute_path, nnx.state(self.latent_step_nnx_model),force=True)

        fol_info(f"{check_point_type} meta flax nnx state is saved to {checkpoint_state_dir}")

    def RestoreState(self,restore_state_directory:str):
        """
        Restores the state of the neural network from a saved checkpoint.

        This method retrieves the saved state of the neural network from a specified directory and updates the model 
        to reflect the restored state.

        Parameters
        ----------
        checkpoint_settings : dict
            A dictionary containing the settings for checkpoint restoration.
            Expected keys:
            - `state_directory` (str): The directory path where the checkpoint is saved.

        Returns
        -------
        None
            The neural network's state is restored and updated in place. A message is logged to confirm the restoration process.

        Notes
        -----
        - Ensure the `state_directory` key is included in the `checkpoint_settings` dictionary, and the specified directory exists.
        - This method uses `nnx.state` to retrieve the current state of the model and updates it with the restored state.
        - Logs the restoration process using `fol_info`.
        """

        # restore nn 
        nn_restore_state_directory = restore_state_directory + "/nn"
        absolute_path = os.path.abspath(nn_restore_state_directory)
        nn_state = nnx.state(self.flax_neural_network)
        restored_state = self.checkpointer.restore(absolute_path, nn_state)
        nnx.update(self.flax_neural_network, restored_state)

        # restore latent 
        latent_restore_state_directory = restore_state_directory + "/latent"
        absolute_path = os.path.abspath(latent_restore_state_directory)
        latent_state = nnx.state(self.latent_step_nnx_model)
        restored_state = self.checkpointer.restore(absolute_path, latent_state)
        nnx.update(self.latent_step_nnx_model, restored_state)

        fol_info(f"meta flax nnx state is restored from {restore_state_directory}")

    def GetState(self):
        return (self.flax_neural_network, self.nnx_optimizer, self.latent_step_nnx_model, self.latent_nnx_optimizer)

    @partial(nnx.jit, static_argnums=(0,))
    def TrainStep(self, meta_state, data):
        nn_model, main_optimizer, latent_step_model, latent_optimizer = meta_state
        batch_loss, meta_grads = nnx.value_and_grad(self.ComputeBatchLossValue,argnums=1) (data,(nn_model,latent_step_model))
        main_optimizer.update(meta_grads[0])
        latent_optimizer.update(meta_grads[1])
        return batch_loss
    
    @partial(nnx.jit, static_argnums=(0,))
    def TestStep(self, meta_state, data):
        nn_model, main_optimizer, latent_step_model, latent_optimizer = meta_state
        return self.ComputeBatchLossValue(data,(nn_model,latent_step_model))

    @print_with_timestamp_and_execution_time
    @partial(nnx.jit, static_argnums=(0,), donate_argnums=1)
    def Predict(self,batch_X:jnp.ndarray):
        preds = self.ComputeBatchPredictions(batch_X,(self.flax_neural_network,self.latent_step_nnx_model))
        return self.loss_function.GetFullDofVector(batch_X,preds.reshape(preds.shape[0], -1))

    @print_with_timestamp_and_execution_time
    @partial(nnx.jit, static_argnums=(0,), donate_argnums=1)
    def PredictDynamics(self,initial_u:jnp.ndarray,num_steps:int):
        """
        Simulates the temporal evolution of the system over multiple time steps using latent loop optimization.

        This method performs sequential predictions starting from an initial state, computing the state at each 
        subsequent time step based on the output of the previous step. At each time step:
        1. A latent code is initialized and optimized to minimize the loss function for the current input state.
        2. The optimized latent code is used to generate the neural network output.
        3. The output is mapped to the full degree of freedom (DoF) vector.
        4. The updated DoF vector becomes the input for the next time step.

        Parameters
        ----------
        initial_u : jnp.ndarray
            The initial condition or state of the system at time step zero. Should be a batch of input vectors.
        
        num_steps : int
            The number of time steps to simulate the system forward in time.

        Returns
        -------
        jnp.ndarray
            An array containing the predicted system states over time.
            The first row corresponds to the initial condition, and each subsequent row corresponds to the predicted 
            state at the next time step. Shape is (num_steps + 1, DoF).

        Notes
        -----
        - Latent code optimization is performed at each time step to generate an accurate prediction of the system's 
        next state.
        - The optimization loop uses `jax.lax.scan` to efficiently perform a fixed number of latent updates.
        - The state update across time steps is performed using `jax.lax.scan` to enable efficient sequential processing.
        - The function uses `jax.vmap` to enable parallel processing of batch inputs at each time step.
        - The model implicitly learns the system dynamics by optimizing the latent representation without requiring 
        explicit temporal modeling.
        - The final output includes both the initial state and the predicted states across all time steps, stacked vertically.
        """ 
        def predict_single_step(sample_u: jnp.ndarray):
            latent_code = jnp.zeros(self.flax_neural_network.in_features)
            control_output = self.control.ComputeControlledVariables(sample_u)
            @jax.jit
            def loss(input_latent_code):
                nn_output = self.flax_neural_network(
                    input_latent_code, self.loss_function.fe_mesh.GetNodesCoordinates()
                ).flatten()[self.loss_function.non_dirichlet_indices]
                return self.loss_function.ComputeSingleLoss(control_output, nn_output)[0]
            loss_latent_grad_fn = jax.grad(loss)
            
            @jax.jit
            def update_latent(latent_code):
                def single_update_latent_fn(state, _):
                    grads = loss_latent_grad_fn(state)
                    update = self.latent_step_nnx_model().value * grads / jnp.linalg.norm(grads)  
                    return state - update, None  
                latent_code, _ = jax.lax.scan(single_update_latent_fn, latent_code, xs=None, length=self.num_latent_iterations)
                return latent_code

            latent_code = update_latent(latent_code)           
            @jax.jit
            def compute_output(latent_code):
                nn_output = self.flax_neural_network(
                    latent_code, self.loss_function.fe_mesh.GetNodesCoordinates()
                ).flatten()[self.loss_function.non_dirichlet_indices]
                return self.loss_function.GetFullDofVector(sample_u, nn_output)      
            return compute_output(latent_code)
        
        parallel_predict_fn = jax.vmap(predict_single_step)

        def scan_fn(u, _):
            u_next = parallel_predict_fn(u)
            return u_next, u_next

        _, dynamic_u = jax.lax.scan(scan_fn, initial_u.reshape(-1,1).T, None, length=num_steps)

        return jnp.vstack((initial_u.reshape(-1,1).T,jnp.squeeze(dynamic_u)))
