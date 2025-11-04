"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: April, 2024
 License: FOL/LICENSE
"""
import os
from abc import ABC,abstractmethod
from typing import Tuple,Iterator
from tqdm import trange
import matplotlib.pyplot as plt
import jax.numpy as jnp
from functools import partial
from flax import nnx
import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import orbax.checkpoint as orbax
from optax import GradientTransformation
import orbax.checkpoint as ocp
from fol.loss_functions.loss import Loss
from fol.tools.decoration_functions import *
from fol.tools.usefull_functions import *

class DeepNetwork(ABC):
    """
    Base abstract class for deep learning models.

    This class serves as a foundation for deep neural networks. It provides a 
    structure to initialize essential components such as the network, optimizer, 
    loss function, and checkpoint settings. The class is abstract and intended 
    to be extended by specific model implementations.

    Attributes:
    ----------
    name : str
        The name of the model, used for identification and checkpointing.
    loss_function : Loss
        The loss function that the model will optimize during training. 
        It defines the objective that the network is learning to minimize.
    flax_neural_network : nnx.Module
        The Flax neural network module that defines the model's architecture.
    optax_optimizer : GradientTransformation
        The Optax optimizer used to update model parameters during training.
    checkpoint_settings : dict, optional
        Dictionary that stores settings for saving and restoring checkpoints. 
        Defaults to an empty dictionary.

    """
    default_convergence_settings = {"num_epochs":100,"convergence_criterion":"total_loss",
                                    "relative_error":1e-8,"absolute_error":1e-8}
    default_plot_settings = {"plot_list":["total_loss"],"plot_frequency":1,"save_frequency":100,"save_directory":"."}
    default_restore_nnx_state_settings = {"restore":False,"state_directory":"flax_state"}
    default_train_checkpoint_settings = {"least_loss_checkpointing":False,"least_loss":np.inf,"frequency":100,"state_directory":"flax_train_state"}
    default_test_checkpoint_settings = {"least_loss_checkpointing":False,"least_loss":np.inf,"frequency":100,"state_directory":"flax_test_state"}
    default_save_nnx_state_settings = {"save_final_state":True,"final_state_directory":"flax_final_state",
                                       "interval_state_checkpointing":False,"interval_state_checkpointing_frequency":0,"interval_state_checkpointing_directory":"."}
    default_data_model_sharding_settings = {"sharding":False,"num_data_devices":1,"num_nnx_model_devices":1}

    def __init__(self,
                 name:str,
                 loss_function:Loss,
                 flax_neural_network:nnx.Module,
                 optax_optimizer:GradientTransformation):
        self.name = name
        self.loss_function = loss_function
        self.flax_neural_network = flax_neural_network
        self.optax_optimizer = optax_optimizer
        self.initialized = False

    def Initialize(self,reinitialize=False) -> None:
        """
        Initialize the deep learning model, its components, and checkpoint settings.

        This method handles the initialization of essential components for the deep network. 
        It ensures that the loss function is initialized, sets up checkpointing 
        for saving and restoring model states, and manages reinitialization if needed. 
        The function is responsible for restoring the model's state from a previous checkpoint, 
        if specified in the checkpoint settings.

        Attributes:
        ----------
        reinitialize : bool, optional
            If True, forces reinitialization of the model and its components even if 
            they have been initialized previously. Default is False.

        Raises:
        -------
        AssertionError:
            If the restored neural network state does not match the current state 
            (based on a comparison using `np.testing.assert_array_equal`).
        """

        # initialize inputs
        if not self.loss_function.initialized:
            self.loss_function.Initialize(reinitialize)

        # create orbax checkpointer
        self.checkpointer = ocp.StandardCheckpointer()

        # initialize the nnx optimizer
        self.nnx_optimizer = nnx.Optimizer(self.flax_neural_network, self.optax_optimizer, wrt=nnx.Param)

    def GetName(self) -> str:
        """
        Returns the name of the model.

        Returns
        -------
        str
            The name of the deep learning model.
        """
        return self.name
    
    @abstractmethod
    def ComputeSingleLossValue(self,x_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Computes the loss value for a single data point.

        This method computes the network's output for a single input data point, 
        applies the control parameters, and evaluates the loss function.

        Parameters
        ----------
        x_set : Tuple[jnp.ndarray, jnp.ndarray]
            A tuple containing the input data and corresponding target labels.
        nn_model : nnx.Module
            The Flax neural network model.

        Returns
        -------
        jnp.ndarray
            The loss value for the single data point.
        """
        pass
    
    @partial(nnx.jit, static_argnums=(0,))
    def ComputeBatchLossValue(self,batch_set:Tuple[jnp.ndarray, jnp.ndarray],nn_model:nnx.Module):
        """
        Computes the loss values for a batch of data.

        This method computes the network's output for a batch of input data, applies the control parameters,
        and evaluates the loss function for the entire batch. It aggregates the results and returns
        summary statistics (min, max, avg) for the batch losses.

        Parameters
        ----------
        batch_set : Tuple[jnp.ndarray, jnp.ndarray]
            A tuple containing a batch of input data and corresponding target labels.
        nn_model : nnx.Module
            The Flax neural network model.

        Returns
        -------
        Tuple[jnp.ndarray, dict]
            The mean loss for the batch and a dictionary of loss statistics (min, max, avg, total).
        """

        batch_losses,(batch_mins,batch_maxs,batch_avgs) = nnx.vmap(self.ComputeSingleLossValue,in_axes=(0, None),out_axes=0)(batch_set,nn_model)
        loss_name = self.loss_function.GetName()
        total_mean_loss = jnp.mean(batch_losses)
        return total_mean_loss, ({loss_name+"_min":jnp.min(batch_mins),
                                         loss_name+"_max":jnp.max(batch_maxs),
                                         loss_name+"_avg":jnp.mean(batch_avgs),
                                         "total_loss":total_mean_loss})

    @partial(nnx.jit, static_argnums=(0,))
    def TrainStep(self, state, data):
        nn, opt = state
        (_,batch_dict), batch_grads = nnx.value_and_grad(self.ComputeBatchLossValue,argnums=1,has_aux=True) \
                                                                    (data,nn)
        opt.update(nn,batch_grads)
        return batch_dict["total_loss"]
    
    @partial(nnx.jit, static_argnums=(0,))
    def TestStep(self, state, data):
        nn,_ = state
        (_,batch_dict) = self.ComputeBatchLossValue(data,nn)
        return batch_dict["total_loss"]

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
              data_model_sharding_settings:dict={},
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

        sharding_settings = UpdateDefaultDict(self.default_data_model_sharding_settings,data_model_sharding_settings)
        fol_info(f"sharding settings:{sharding_settings}")

        # restore state if needed 
        if restore_nnx_state_settings['restore']:
            self.RestoreState(restore_nnx_state_settings["state_directory"])

        # adjust batch for parallization reasons
        adjusted_batch_size = next(i for i in range(batch_size, 0, -1) if len(train_set[0]) % i == 0)   
        if adjusted_batch_size!=batch_size:
            fol_info(f"for the parallelization of batching, the batch size is changed from {batch_size} to {adjusted_batch_size}")   
            batch_size = adjusted_batch_size  

        # sharding & data-model parallelization
        if sharding_settings["sharding"]:
            num_data_devices = sharding_settings["num_data_devices"]
            num_model_devices = sharding_settings["num_nnx_model_devices"]
            if num_data_devices * num_model_devices != jax.local_device_count():
                fol_error(f"number of available devices (i.e., {jax.local_device_count()}) does not match with the mutiplication of number of data and model devices (i.e., {(num_data_devices,num_model_devices)}) !")

            if len(train_set[0]) % num_data_devices != 0:
                fol_error(f"size/shape of train_set (i.e., {train_set[0].shape}) is not a multiplier of data devices (i.e.,{num_data_devices}) for sharding !")

            if len(test_set)>0:
                if len(test_set[0]) % num_data_devices != 0:
                    fol_error(f"size/shape of test_set (i.e., {test_set[0].shape}) is not a multiplier of data devices (i.e.,{num_data_devices}) for sharding !")

            sharding_mesh = jax.sharding.Mesh(devices=np.array(jax.devices()).reshape(num_data_devices, num_model_devices),
                                                axis_names=('data', 'model'))

            nnx_model_sharding = jax.NamedSharding(sharding_mesh, jax.sharding.PartitionSpec('model'))
            data_sharding = jax.NamedSharding(sharding_mesh, jax.sharding.PartitionSpec('data'))

            # data sharding
            train_set = jax.device_put(train_set, data_sharding)
                
            if len(test_set)>0:
                test_set = jax.device_put(test_set, data_sharding)

            # nnx model sharding
            with sharding_mesh:
                state = nnx.state(self.flax_neural_network)   
                pspecs = nnx.get_partition_spec(state)
                sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
                nnx.update(self.flax_neural_network, sharded_state) 

            fol_info("neural network is sharded as ")
            jax.debug.visualize_array_sharding(self.flax_neural_network.synthesizer_nn.nn_params[0][0])
            fol_info("train set is sharded as ")
            jax.debug.visualize_array_sharding(train_set[0])
            if len(test_set)>0:
                fol_info("test set is sharded as ")
                jax.debug.visualize_array_sharding(test_set[0])                

        def train_loop():

            train_history_dict = {"total_loss":[]}
            test_history_dict = {"total_loss":[]}
            pbar = trange(convergence_settings["num_epochs"])
            converged = False
            rng, _ = jax.random.split(jax.random.PRNGKey(0))

            state = (self.flax_neural_network, self.nnx_optimizer)

            # Most powerful chicken seasoning taken from https://gist.github.com/puct9/35bb1e1cdf9b757b7d1be60d51a2082b 
            # and discussions in https://github.com/google/flax/issues/4045
            train_multiple_steps_with_idxs = nnx.jit(lambda st, dat, idxs: nnx.scan(lambda st, idxs: (st, self.TrainStep(st, jax.tree.map(lambda a: a[idxs], dat))))(st, idxs))    

            for epoch in pbar:
                # update least values in case of restore
                if train_checkpoint_settings["least_loss_checkpointing"] and restore_nnx_state_settings['restore'] and epoch==0:
                    train_checkpoint_settings["least_loss"] = self.TestStep(state,train_set)
                if test_checkpoint_settings["least_loss_checkpointing"] and restore_nnx_state_settings['restore'] and epoch==0:
                    test_checkpoint_settings["least_loss"] = self.TestStep(state,test_set)            

                # parallel batching and train step
                rng, sub = jax.random.split(rng)
                order = jax.random.permutation(sub, len(train_set[0])).reshape(-1, batch_size)
                _, losses = train_multiple_steps_with_idxs(state, train_set, order)
                train_history_dict["total_loss"].append(losses.mean())
                
                # test step
                if len(test_set[0])>0 and ((epoch)%test_frequency==0 or epoch==convergence_settings["num_epochs"]-1):
                    test_history_dict["total_loss"].append(self.TestStep(state,test_set))
                
                # print step   
                if len(test_set[0])>0:
                    print_dict = {"train_loss":train_history_dict["total_loss"][-1],
                                "test_loss":test_history_dict["total_loss"][-1]}
                else:
                    print_dict = {"train_loss":train_history_dict["total_loss"][-1]}

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


        if sharding_settings["sharding"]:
            with sharding_mesh:
                train_loop()
        else:
            train_loop()
            
    def CheckConvergence(self,train_history_dict:dict,convergence_settings:dict):
        """
        Checks whether the training process has converged.

        This method evaluates the training history based on the defined convergence 
        criterion, absolute error, or relative error. If the conditions are met, 
        it returns True, indicating convergence.

        Parameters
        ----------
        train_history_dict : dict
            The history of the training loss values.
        convergence_settings : dict
            The settings that define when convergence occurs, including absolute error 
            and relative error thresholds.

        Returns
        -------
        bool
            True if the model has converged, False otherwise.
        """
        convergence_criterion = convergence_settings["convergence_criterion"]
        absolute_error = convergence_settings["absolute_error"]
        relative_error = convergence_settings["relative_error"]
        num_epochs = convergence_settings["num_epochs"]
        current_epoch = len(train_history_dict[convergence_criterion])
        # check for absolute and relative errors and convergence
        if abs(train_history_dict[convergence_criterion][-1])<absolute_error:
            return True
        if current_epoch>1:
            if abs(train_history_dict[convergence_criterion][-1] -
                   train_history_dict[convergence_criterion][-2])<relative_error:
                return True
            elif current_epoch>=num_epochs:
                return True
            else:
                return False
        else:
            return False   

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

        absolute_path = os.path.abspath(restore_state_directory)
        # get the state
        nn_state = nnx.state(self.flax_neural_network)
        # restore
        restored_state = self.checkpointer.restore(absolute_path, nn_state)
        # now update the model with the loaded state
        nnx.update(self.flax_neural_network, restored_state)
        fol_info(f"flax nnx state is restored from {restore_state_directory}")

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

        absolute_path = os.path.abspath(checkpoint_state_dir)
        self.checkpointer.save(absolute_path, nnx.state(self.flax_neural_network),force=True)
        fol_info(f"{check_point_type} flax nnx state is saved to {checkpoint_state_dir}")

    def PlotHistoryDict(self,plot_settings:dict,train_history_dict:dict,test_history_dict:dict):
        """
        Plots the training and testing history.

        This method generates and saves a plot of the training and test history based on 
        the specified settings. It supports logging various metrics, such as loss, 
        across training epochs and allows customization of which metrics to plot.

        Parameters
        ----------
        plot_settings : dict
            Dictionary containing settings for the plot, such as:
            - 'plot_rate': int, how often to plot the history (in terms of epochs).
            - 'plot_list': list of str, the metrics to be plotted (e.g., 'total_loss').
        train_history_dict : dict
            A dictionary where keys are metric names (e.g., 'total_loss') and values 
            are lists of the corresponding metric values during training.
        test_history_dict : dict
            A dictionary where keys are metric names (e.g., 'total_loss') and values 
            are lists of the corresponding metric values during testing.

        Returns
        -------
        None
            The function does not return any values but saves the plot to a file 
            in the working directory as 'training_history.png'.
        """
        plot_rate = plot_settings["plot_frequency"]
        plot_list = plot_settings["plot_list"]

        plt.figure(figsize=(10, 5))
        train_max_length = 0
        for key,value in train_history_dict.items():
            if len(value)>0 and (len(plot_list)==0 or key in plot_list):
                train_max_length = len(value)
                plt.semilogy(value[::plot_rate], label=f"train_{key}") 

        for key,value in test_history_dict.items():
            if len(value)>0 and (len(plot_list)==0 or key in plot_list):
                test_length = len(value)
                x_value = [ i * plot_settings["test_frequency"] for i in range(test_length-1)]
                x_value.append(train_max_length-1)
                plt.semilogy(x_value,value[::plot_rate], label=f"test_{key}") 

        plt.title("Training History")
        plt.xlabel(str(plot_rate) + " Epoch")
        plt.ylabel("Log Value")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_settings["save_directory"],"training_history.png"), bbox_inches='tight')
        plt.close()

    @abstractmethod
    def Finalize(self) -> None:
        """Finalizes the network.

        This method finalizes the network. This is only called once in the whole training process.

        """
        pass





