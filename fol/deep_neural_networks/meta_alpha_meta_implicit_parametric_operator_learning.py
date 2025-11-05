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
        self.latent_nnx_optimizer = nnx.Optimizer(self.latent_step_nnx_model,self.latent_step_optimizer, wrt=nnx.Param)

    @partial(nnx.jit, static_argnums=(0,))
    def ComputeSingleLossValue(self,latent_and_control:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        latent_code = latent_and_control[0]
        control_output = self.control.ComputeControlledVariables(latent_and_control[1])
        nn_output = nn_model(latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
        return self.loss_function.ComputeSingleLoss(control_output,nn_output)

    def Finalize(self):
        pass
    
    @partial(nnx.jit, static_argnums=(0,))
    def ComputeBatchLatent(self,batch_X:jnp.ndarray,flax_neural_network:nnx.Module,latent_step:nnx.Module):
        @nnx.jit
        def compute_single_latent(sample_x:jnp.ndarray):

            latent_code = jnp.zeros(flax_neural_network.in_features)
            control_output = self.control.ComputeControlledVariables(sample_x)

            @nnx.jit
            def loss(input_latent_code):
                nn_output = flax_neural_network(input_latent_code,self.loss_function.fe_mesh.GetNodesCoordinates()).flatten()[self.loss_function.non_dirichlet_indices]
                return self.loss_function.ComputeSingleLoss(control_output,nn_output)[0]

            loss_latent_grad_fn = jax.grad(loss)
            for _ in range(self.num_latent_iterations):
                latent_code -= latent_step() * loss_latent_grad_fn(latent_code)

            return latent_code

        return jnp.array(jax.vmap(compute_single_latent)(batch_X))

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

    @partial(nnx.jit, static_argnums=(0,))
    def TrainStep(self, meta_state, data):
        nn_model, main_optimizer, latent_step_model, latent_optimizer = meta_state
        meta_model = (nn_model,latent_step_model)

        @nnx.jit
        def compute_batch_loss(batch_X,meta_model):
            nn_model, latent_step_model = meta_model
            latent_codes = self.ComputeBatchLatent(batch_X,nn_model,latent_step_model)
            return self.ComputeBatchLossValue((latent_codes,batch_X),nn_model)[0],latent_codes
        
        (loss_value,latent_codes),meta_grads = nnx.value_and_grad(compute_batch_loss,argnums=1,has_aux=True) (data[0],meta_model)
        main_optimizer.update(nn_model,meta_grads[0])
        latent_optimizer.update(latent_step_model,meta_grads[1])
        return loss_value
    
    @partial(nnx.jit, static_argnums=(0,))
    def TestStep(self, meta_state, data):
        nn_model, main_optimizer, latent_step_model, latent_optimizer = meta_state
        latent_codes = self.ComputeBatchLatent(data[0],nn_model,latent_step_model)
        return self.ComputeBatchLossValue((latent_codes,data[0]),nn_model)[0]

    @print_with_timestamp_and_execution_time
    def Train(self, 
              train_set:Tuple[jnp.ndarray, jnp.ndarray], 
              test_set:Tuple[jnp.ndarray, jnp.ndarray] = (jnp.array([]), jnp.array([])),
              test_frequency:int=100, 
              batch_size:int=100, 
              convergence_settings:dict={}, 
              plot_settings:dict={},
              restore_nnx_state_settings:dict={},
              train_checkpoint_settings:dict={},
              test_checkpoint_settings:dict={},
              save_nnx_state_settings:dict={},
              working_directory='.'):

        convergence_settings = UpdateDefaultDict(self.default_convergence_settings,convergence_settings)
        fol_info(f"convergence settings:{convergence_settings}")

        default_plot_settings = copy.deepcopy(self.default_plot_settings)
        default_plot_settings["save_directory"] = working_directory
        plot_settings = UpdateDefaultDict(default_plot_settings,plot_settings)
        plot_settings["test_frequency"] = test_frequency
        fol_info(f"plot settings:{plot_settings}")

        default_restore_nnx_state_settings = copy.deepcopy(self.default_restore_nnx_state_settings)
        default_restore_nnx_state_settings["state_directory"] = working_directory + "/" + default_restore_nnx_state_settings["state_directory"]
        restore_nnx_state_settings = UpdateDefaultDict(default_restore_nnx_state_settings,restore_nnx_state_settings)
        fol_info(f"restore settings:{restore_nnx_state_settings}")

        default_train_checkpoint_settings = copy.deepcopy(self.default_train_checkpoint_settings)
        default_train_checkpoint_settings["state_directory"] = working_directory + "/" + default_train_checkpoint_settings["state_directory"]
        train_checkpoint_settings = UpdateDefaultDict(default_train_checkpoint_settings,train_checkpoint_settings)
        fol_info(f"train checkpoint settings:{train_checkpoint_settings}")

        default_test_checkpoint_settings = copy.deepcopy(self.default_test_checkpoint_settings)
        default_test_checkpoint_settings["state_directory"] = working_directory + "/" + default_test_checkpoint_settings["state_directory"]
        test_checkpoint_settings = UpdateDefaultDict(default_test_checkpoint_settings,test_checkpoint_settings)
        fol_info(f"test checkpoint settings:{test_checkpoint_settings}")
        
        default_save_nnx_state_settings = copy.deepcopy(self.default_save_nnx_state_settings)
        default_save_nnx_state_settings["final_state_directory"] = working_directory + "/" + default_save_nnx_state_settings["final_state_directory"]
        default_save_nnx_state_settings["interval_state_checkpointing_directory"] = working_directory + "/" + default_save_nnx_state_settings["interval_state_checkpointing_directory"]
        save_nnx_state_settings = UpdateDefaultDict(default_save_nnx_state_settings,save_nnx_state_settings)
        fol_info(f"save nnx state settings:{save_nnx_state_settings}")

        # restore state if needed 
        if restore_nnx_state_settings['restore']:
            self.RestoreState(restore_nnx_state_settings["state_directory"])

        # adjust batch for parallization reasons
        adjusted_batch_size = next(i for i in range(batch_size, 0, -1) if len(train_set[0]) % i == 0)   
        if adjusted_batch_size!=batch_size:
            fol_info(f"for the parallelization of batching, the batch size is changed from {batch_size} to {adjusted_batch_size}")   
            batch_size = adjusted_batch_size  

        train_history_dict = {"total_loss":[]}
        test_history_dict = {"total_loss":[]}
        pbar = trange(convergence_settings["num_epochs"])
        converged = False
        rng, _ = jax.random.split(jax.random.PRNGKey(0))
        meta_state = (self.flax_neural_network, self.nnx_optimizer, self.latent_step_nnx_model, self.latent_nnx_optimizer)

        # Most powerful chicken seasoning taken from https://gist.github.com/puct9/35bb1e1cdf9b757b7d1be60d51a2082b 
        # and discussions in https://github.com/google/flax/issues/4045
        train_multiple_steps_with_idxs = nnx.jit(lambda st, dat, idxs: nnx.scan(lambda st, idxs: (st, self.TrainStep(st, jax.tree.map(lambda a: a[idxs], dat))))(st, idxs))    

        for epoch in pbar:
            # update least values in case of restore
            if train_checkpoint_settings["least_loss_checkpointing"] and restore_nnx_state_settings['restore'] and epoch==0:
                train_checkpoint_settings["least_loss"] = self.TestStep(meta_state,train_set)
            if test_checkpoint_settings["least_loss_checkpointing"] and restore_nnx_state_settings['restore'] and epoch==0:
                test_checkpoint_settings["least_loss"] = self.TestStep(meta_state,test_set)            

            # parallel batching and train step
            rng, sub = jax.random.split(rng)
            order = jax.random.permutation(sub, len(train_set[0])).reshape(-1, batch_size)            
            _, losses = train_multiple_steps_with_idxs(meta_state, train_set, order)
            train_history_dict["total_loss"].append(losses.mean())
            
            # test step
            if len(test_set[0])>0 and ((epoch)%test_frequency==0 or epoch==convergence_settings["num_epochs"]-1):
                test_history_dict["total_loss"].append(self.TestStep(meta_state,test_set))
            
            # print step   
            if len(test_set[0])>0:
                print_dict = {"train_loss":train_history_dict["total_loss"][-1],
                              "test_loss":test_history_dict["total_loss"][-1],
                              "latent_step":self.latent_step_nnx_model().value}
            else:
                print_dict = {"train_loss":train_history_dict["total_loss"][-1],
                              "latent_step":self.latent_step_nnx_model().value}

            pbar.set_postfix(print_dict)

            # check converged
            converged = self.CheckConvergence(train_history_dict,convergence_settings)

            # plot the histories
            if (epoch>0 and epoch %plot_settings["save_frequency"] == 0) or converged:
                self.PlotHistoryDict(plot_settings,train_history_dict,test_history_dict)

            # train checkpointing
            if train_checkpoint_settings["least_loss_checkpointing"] and epoch>0 and \
                (epoch)%train_checkpoint_settings["frequency"] == 0 and \
                train_history_dict["total_loss"][-1] < train_checkpoint_settings["least_loss"]:
                fol_info(f"train total_loss improved from {train_checkpoint_settings['least_loss']} to {train_history_dict['total_loss'][-1]}")
                train_checkpoint_settings["least_loss"] = train_history_dict["total_loss"][-1]
                self.SaveCheckPoint("train",train_checkpoint_settings["state_directory"])

            # test checkpointing
            if test_checkpoint_settings["least_loss_checkpointing"] and epoch>0 and \
                (epoch)%test_checkpoint_settings["frequency"] == 0 and \
                test_history_dict["total_loss"][-1] < test_checkpoint_settings["least_loss"]:
                fol_info(f"test total_loss improved from {test_checkpoint_settings['least_loss']} to {test_history_dict['total_loss'][-1]}")
                test_checkpoint_settings["least_loss"] = test_history_dict["total_loss"][-1]
                self.SaveCheckPoint("test",test_checkpoint_settings["state_directory"])

            # interval checkpointing
            if save_nnx_state_settings["interval_state_checkpointing"] and epoch>0 and \
            (epoch)%save_nnx_state_settings["interval_state_checkpointing_frequency"] == 0:
                self.SaveCheckPoint(f"interval {epoch}",save_nnx_state_settings["interval_state_checkpointing_directory"]+"/flax_train_state_epoch_"+str(epoch))

            if epoch<convergence_settings["num_epochs"]-1 and converged:
                break          

        if train_checkpoint_settings["least_loss_checkpointing"] and \
            train_history_dict["total_loss"][-1] < train_checkpoint_settings['least_loss']:
            fol_info(f"train total_loss improved from {train_checkpoint_settings['least_loss']} to {train_history_dict['total_loss'][-1]}")
            self.SaveCheckPoint("train",train_checkpoint_settings["state_directory"])

        if test_checkpoint_settings["least_loss_checkpointing"] and \
            test_history_dict["total_loss"][-1] < test_checkpoint_settings['least_loss']:
            fol_info(f"test total_loss improved from {test_checkpoint_settings['least_loss']} to {test_history_dict['total_loss'][-1]}")
            self.SaveCheckPoint("test",test_checkpoint_settings["state_directory"])

        if save_nnx_state_settings["save_final_state"]:
            self.SaveCheckPoint("final",save_nnx_state_settings["final_state_directory"])

        self.checkpointer.close()  # Close resources properly

    @print_with_timestamp_and_execution_time
    def Predict(self,batch_control:jnp.ndarray):
        batch_X = jax.vmap(self.control.ComputeControlledVariables)(batch_control)
        latent_codes = self.ComputeBatchLatent(batch_control,self.flax_neural_network,self.latent_step_nnx_model)
        batch_Y =jax.vmap(self.flax_neural_network,(0,None))(latent_codes,self.loss_function.fe_mesh.GetNodesCoordinates())
        batch_Y = batch_Y.reshape(latent_codes.shape[0], -1)[:,self.loss_function.non_dirichlet_indices]
        return jax.vmap(self.loss_function.GetFullDofVector)(batch_X,batch_Y)

    @print_with_timestamp_and_execution_time
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
