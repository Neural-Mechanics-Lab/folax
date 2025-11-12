"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: November, 2024
 License: FOL/LICENSE
"""
import copy
from flax import nnx
from jax.nn import relu,sigmoid,swish,tanh,leaky_relu,elu
from jax.numpy import sin
import jax
import jax.numpy as jnp
from jax._src.typing import Array, ArrayLike
from jax import random
from fol.tools.usefull_functions import *
from fol.tools.decoration_functions import *

def layer_init_factopry(key:Array,
                        in_dim:int,
                        out_dim:int,
                        activation_settings:dict):
    
    """
    Initializes weights and biases for a layer based on activation settings.

    Args:
        key (Array): PRNG key for random initialization.
        in_dim (int): Number of input features.
        out_dim (int): Number of output features.
        activation_settings (dict): Dictionary containing activation configuration, including type.

    Returns:
        Tuple[Array, Array]: Initialized weights and biases.
    """

    if activation_settings["type"]=="sin":
        return siren_init(key,in_dim,out_dim,activation_settings)
    else:
        if activation_settings["type"] in ["relu","leaky_relu","elu"]:
            init_weights = nnx.initializers.he_uniform()(key,(in_dim,out_dim))
        elif activation_settings["type"] == "tanh":
            init_weights = nnx.initializers.glorot_uniform()(key,(in_dim,out_dim))
        else:
            init_weights = nnx.initializers.lecun_uniform()(key,(in_dim,out_dim))
        init_biases = nnx.initializers.zeros(key,(out_dim,))

        return init_weights,init_biases

def siren_init(key:Array,in_dim:int,out_dim:int,activation_settings:dict):
    """
    Custom initialization method for SIREN layers.

    This initialization method is designed to support sinusoidal representation networks (SIRENs), 
    which use periodic activation functions. The approach is inspired by the following papers:

    - Sitzmann, V., Martel, J., Bergman, A., Lindell, D., & Wetzstein, G. (2020). 
      Implicit neural representations with periodic activation functions. 
      Advances in Neural Information Processing Systems, 33, 7462-7473.
    - Yeom, T., Lee, S., & Lee, J. (2024). Fast Training of Sinusoidal Neural Fields 
      via Scaling Initialization. arXiv preprint arXiv:2410.04779.

    Args:
        key (Array): PRNG key for random initialization.
        in_dim (int): Number of input features.
        out_dim (int): Number of output features.
        activation_settings (dict): Dictionary containing SIREN-specific initialization parameters:
            - "current_layer_idx": Index of the current layer.
            - "total_num_layers": Total number of layers.
            - "initialization_gain": Weight scale for initialization.
            - "prediction_gain": Omega factor for SIREN layers.

    Returns:
        Tuple[Array, Array]: Initialized weights and biases.
    """

    weight_key, bias_key = random.split(key)
    current_layer_idx = activation_settings["current_layer_idx"]
    total_num_layers = activation_settings["total_num_layers"]
    weight_scale = activation_settings["initialization_gain"]
    omega = activation_settings["prediction_gain"]

    if current_layer_idx == 0: weight_variance = weight_scale / in_dim
    elif current_layer_idx == total_num_layers-2: weight_variance = jnp.sqrt(6 / in_dim) / omega
    else: weight_variance = weight_scale * jnp.sqrt(6 / in_dim) / omega
    
    init_weights = random.uniform(weight_key, (in_dim, out_dim), jnp.float32, minval=-weight_variance, maxval=weight_variance)
    init_biases = jnp.zeros(out_dim)
    return init_weights,init_biases

class MLP(nnx.Module):
    """
    A multi-layer perceptron (MLP) with customizable activation functions, skip connections, 
    and initialization strategies.

    Args:
        name (str): Name of the MLP instance.
        input_size (int): Number of input features.
        output_size (int): Number of output features.
        hidden_layers (list): List of integers specifying the number of units in each hidden layer.
        activation_settings (dict, optional): Configuration for activation functions. Defaults to:
            - "type": Activation type (e.g., "sin", "relu", etc.).
            - "prediction_gain": Gain for scaling activations (default 30 for "sin").
            - "initialization_gain": Gain for weight initialization (default 1).
        use_bias (bool, optional): Whether to include biases in the layers. Defaults to True.
        skip_connections_settings (dict, optional): Configuration for skip connections. Defaults to:
            - "active": Whether to enable skip connections.
            - "frequency": Frequency of skip connections (in layers).

    Attributes:
        name (str): Name of the MLP instance.
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        hidden_layers (list): Configuration of hidden layers.
        activation_settings (dict): Activation function settings.
        use_bias (bool): Whether biases are used in the layers.
        skip_connections_settings (dict): Skip connection configuration.
        nn_params (list): List of network parameters (weights and biases).
        act_func (Callable): Activation function used in the MLP.
        act_func_gain (float): Gain applied to activation function outputs.
        fw_func (Callable): Forward pass function (with or without skip connections).
        total_num_weights (int): Total number of weights in the network.
        total_num_biases (int): Total number of biases in the network.
    """
    @print_with_timestamp_and_execution_time
    def __init__(self,name:str,  
                      input_size:int=0,
                      output_size: int=0, 
                      hidden_layers:list=[],
                      activation_settings:dict={},
                      use_bias:bool=True,
                      skip_connections_settings:dict={}):
        """
        Initializes the MLP with specified parameters and configurations.

        Args:
            name (str): Name of the MLP instance.
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            hidden_layers (list): List specifying hidden layer sizes.
            activation_settings (dict): Configuration for activation functions.
            use_bias (bool): Whether to include biases in the layers.
            skip_connections_settings (dict): Configuration for skip connections.
        """
        self.name = name
        self.in_features=input_size
        self.out_features=output_size
        self.hidden_layers = hidden_layers
        self.activation_settings = activation_settings
        self.use_bias = use_bias
        self.skip_connections_settings = skip_connections_settings

        default_activation_settings={"type":"sin",
                                    "prediction_gain":30,
                                    "initialization_gain":1}
        self.activation_settings = UpdateDefaultDict(default_activation_settings,
                                                     self.activation_settings)
        
        default_skip_connections_settings = {"active":False,"frequency":1}
        self.skip_connections_settings = UpdateDefaultDict(default_skip_connections_settings,
                                                            self.skip_connections_settings) 

        self.InitialNetworkParameters()
        fol_info(f"MLP network is initialized by {self.total_num_weights} weights and {self.total_num_biases} biases !")

        act_name = self.activation_settings["type"]
        self.act_func = globals()[act_name]
        if act_name=="sin":
            self.act_func_gain = self.activation_settings["prediction_gain"]
        else:
            self.act_func_gain = 1 
        
        if self.skip_connections_settings["active"]:
            self.fw_func = self.ForwardSkip
        else:
            self.fw_func = self.Forward

    def InitialNetworkParameters(self):
        """
        Initializes the weights and biases of the MLP based on layer sizes and 
        activation settings, with support for skip connections.
        """
        layer_sizes = [self.in_features] +  self.hidden_layers
        if self.out_features != 0:
            layer_sizes += [self.out_features]

        activation_settings = copy.deepcopy(self.activation_settings)
        activation_settings["total_num_layers"] = len(layer_sizes)

        key = random.PRNGKey(0)
        keys = random.split(key, len(layer_sizes) - 1)

        self.nn_params = nnx.List([])
        self.total_num_weights = 0
        self.total_num_biases = 0
        for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            activation_settings["current_layer_idx"] = i
            if self.skip_connections_settings["active"] and i>0 and \
                (i%self.skip_connections_settings["frequency"]==0):
                init_weights,init_biases = layer_init_factopry(keys[i],in_dim+self.in_features,out_dim,activation_settings)
            else:
                init_weights,init_biases = layer_init_factopry(keys[i],in_dim,out_dim,activation_settings)
            
            self.total_num_weights += init_weights.size
            if self.use_bias:
                self.nn_params.append((nnx.Param(init_weights),nnx.Param(init_biases)))
                self.total_num_biases += init_biases.size
            else:
                self.nn_params.append((nnx.Param(init_weights),nnx.Variable(jnp.zeros_like(init_biases))))

    def GetName(self):
        """
        Retrieves the name of the MLP instance.

        Returns:
            str: Name of the MLP instance.
        """
        return self.name

    def ComputeX(self,w:nnx.Param,prev_x:jax.Array,b:nnx.Param):
        """
        Computes the output of a layer without skip connections.

        Args:
            w (nnx.Param): Weight matrix.
            prev_x (jax.Array): Input to the layer.
            b (nnx.Param): Bias vector.

        Returns:
            jax.Array: Output of the layer.
        """
        return prev_x @ w + b
    
    def Forward(self,x: jax.Array,nn_params:list[tuple[nnx.Param, nnx.Param]]):
        """
        Performs a forward pass through the MLP without skip connections.

        Args:
            x (jax.Array): Input to the network.
            nn_params (list[tuple[nnx.Param, nnx.Param]]): List of layer parameters (weights and biases).

        Returns:
            jax.Array: Output of the network.
        """
        for (w, b) in nn_params[:-1]:
            x = self.ComputeX(w,x,b)
            x = self.act_func(self.act_func_gain*x)
        final_w, final_b = nn_params[-1]
        return self.ComputeX(final_w,x,final_b)
    
    def ComputeXSkip(self,w:nnx.Param,prev_x:jax.Array,in_x:jax.Array,b:nnx.Param):
        """
        Computes the output of a layer with skip connections.

        Args:
            w (nnx.Param): Weight matrix.
            prev_x (jax.Array): Input to the current layer.
            in_x (jax.Array): Original input to the network for skip connection.
            b (nnx.Param): Bias vector.

        Returns:
            jax.Array: Output of the layer.
        """
        return jnp.hstack((prev_x,in_x.copy())) @ w + b
    
    def ForwardSkip(self,x:jax.Array,nn_params:list[tuple[nnx.Param, nnx.Param]]):
        """
        Performs a forward pass through the MLP with skip connections.

        Args:
            x (jax.Array): Input to the network.
            nn_params (list[tuple[nnx.Param, nnx.Param]]): List of layer parameters (weights and biases).

        Returns:
            jax.Array: Output of the network.
        """
        in_x = x.copy()
        layer_num = 0
        for (w, b) in nn_params[0:-1]:
            if layer_num>0 and layer_num%self.skip_connections_settings["frequency"]==0:
                x = self.ComputeXSkip(w,x,in_x,b)
            else:
                x = self.ComputeX(w,x,b)
            x = self.act_func(self.act_func_gain*x)
            layer_num += 1

        final_w, final_b = nn_params[-1]

        if layer_num%self.skip_connections_settings["frequency"]==0:
            return self.ComputeXSkip(final_w,x,in_x,final_b)
        else:
            return self.ComputeX(final_w,x,final_b)

    def __call__(self, x: jax.Array):
        """
        Executes a forward pass through the MLP using the configured forward method.

        Args:
            x (jax.Array): Input to the network.

        Returns:
            jax.Array: Output of the network.
        """
        return self.fw_func(x,self.nn_params)

class HyperNetwork(nnx.Module):
    """
    A configurable hypernetwork that integrates a modulator network and a synthesizer 
    network with support for various coupling mechanisms. 

    The hypernetwork enables dynamic parameter generation and modulation for the synthesizer 
    network based on the output of the modulator network. It is designed for tasks requiring 
    flexible and adaptive parameter sharing between neural networks.

    Attributes:
        name (str): Name of the hypernetwork module.
        modulator_nn (MLP): The modulator neural network responsible for generating parameters 
            that influence the synthesizer network.
        synthesizer_nn (MLP): The synthesizer neural network responsible for task-specific 
            computations.
        in_features (int): Number of input features for the modulator network.
        out_features (int): Number of output features for the synthesizer network.
        coupling_settings (dict): A dictionary specifying the coupling mechanism configuration. 
            Includes:
            - "coupled_variable": Specifies the variable to couple (default is "shift").
            - "modulator_to_synthesizer_coupling_mode": Defines the coupling mode between 
              modulator and synthesizer networks. Options: "all_to_all", "last_to_all", 
              "one_modulator_per_synthesizer_layer".
        total_num_weights (int): Total number of weights in the combined hypernetwork.
        total_num_biases (int): Total number of biases in the combined hypernetwork.
        fw_func (function): The forward propagation function, determined by the selected 
            coupling mode.

    Coupling Modes:
        - "all_to_all": Every layer of the modulator network is coupled to every layer of the 
          synthesizer network. Requires identical hidden layer configurations in both networks.
        - "last_to_all": The last layer of the modulator network is coupled to all layers of 
          the synthesizer network.
        - "one_modulator_per_synthesizer_layer": Each layer of the synthesizer network is 
          modulated by a dedicated modulator network.

    Raises:
        ValueError: If invalid or unsupported coupling settings are provided.
    """
    @print_with_timestamp_and_execution_time
    def __init__(self,name:str,
                      modulator_nn:MLP,
                      synthesizer_nn:MLP,
                      coupling_settings:dict={}):
        self.name = name
        self.modulator_nn = modulator_nn
        self.synthesizer_nn = synthesizer_nn

        self.in_features = self.modulator_nn.in_features
        self.out_features = self.synthesizer_nn.out_features
        
        self.coupling_settings = {"coupled_variable":"shift",
                                  "modulator_to_synthesizer_coupling_mode":"all_to_all"} # other coupling options: last_to_all,last_to_last                  

        self.coupling_settings = UpdateDefaultDict(self.coupling_settings,coupling_settings)

        if self.coupling_settings["coupled_variable"] != "shift":
            coupled_variable = self.coupling_settings["coupled_variable"]
            fol_error(f"coupled_variable {coupled_variable} is not supported, options are shift")

        if self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "all_to_all":
            if self.modulator_nn.hidden_layers != self.synthesizer_nn.hidden_layers:
                fol_error(f"for all_to_all modulator to synthesizer coupling, hidden layers of synthesizer and modulator NNs should be identical !")
            self.fw_func = self.all_to_all_fw
            self.total_num_weights = self.modulator_nn.total_num_weights + \
                                    self.synthesizer_nn.total_num_weights
            self.total_num_biases = self.modulator_nn.total_num_biases +\
                                    self.synthesizer_nn.total_num_biases
        elif self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "last_to_all":
            synthesizer_modulated_biases = self.synthesizer_nn.total_num_biases
            synthesizer_modulated_biases -= self.synthesizer_nn.out_features # subtract the last linear layer
            modulator_original_out_features = self.modulator_nn.out_features

            self.modulator_nn.out_features = synthesizer_modulated_biases
            fol_info(f" the out_features of modulator network is changed from {modulator_original_out_features} to \
                        the total number of the modulated biases of the synthesizer network {synthesizer_modulated_biases}")
            
            self.modulator_nn.InitialNetworkParameters()
            fol_info(f"the modulator network is re-initialized by {self.modulator_nn.total_num_weights} weights and {self.modulator_nn.total_num_biases} biases !")
            self.fw_func = self.last_to_all_fw
            self.total_num_weights = self.modulator_nn.total_num_weights + \
                                    self.synthesizer_nn.total_num_weights
            self.total_num_biases = self.modulator_nn.total_num_biases + \
                                    self.synthesizer_nn.total_num_biases
        elif self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "one_modulator_per_synthesizer_layer":
            self.total_num_biases = self.synthesizer_nn.total_num_biases
            self.total_num_weights = self.synthesizer_nn.total_num_weights
            self.modulator_nns = []
            for i in range(len(self.synthesizer_nn.hidden_layers)):
                synthesizer_layer_biases = self.synthesizer_nn.hidden_layers[i]
                synthesizer_layer_modulator = MLP(name=f"synthesizer_layer_{i}_modulator",
                                                  input_size=self.modulator_nn.in_features,
                                                  output_size=synthesizer_layer_biases,
                                                  hidden_layers=self.modulator_nn.hidden_layers,
                                                  activation_settings=self.modulator_nn.activation_settings,
                                                  use_bias=self.modulator_nn.use_bias,
                                                  skip_connections_settings=self.modulator_nn.skip_connections_settings)
                self.modulator_nns.append(synthesizer_layer_modulator)
                self.total_num_biases += synthesizer_layer_modulator.total_num_biases
                self.total_num_weights += synthesizer_layer_modulator.total_num_weights

            fol_info(f" created {len(self.synthesizer_nn.hidden_layers)} modulators, i.e., one modulator per synthesizer layer !")
            # delete modulator_nn
            del self.modulator_nn
            # set fw function
            self.fw_func = self.one_modulator_per_synthesizer_layer_fw
        else:
            valid_options=["all_to_all","last_to_all","one_modulator_per_synthesizer_layer"]
            fol_error(f"valid options for modulator_to_synthesizer_coupling_mode are {valid_options} !")

        fol_info(f"hyper network has {self.total_num_weights} weights and {self.total_num_biases} biases !")

    def GetName(self):
        """
        Retrieves the name of the hypernetwork module.

        Returns:
            str: The name of the hypernetwork module.
        """
        return self.name

    def all_to_all_fw(self,latent_array:jax.Array,coord_matrix:jax.Array,
                            modulator_nn:MLP,synthesizer_nn:MLP):
        """
        Implements the "all-to-all" forward propagation coupling mechanism.

        In this coupling mode, each layer of the modulator network is coupled with 
        the corresponding layer of the synthesizer network. This mode requires that 
        the modulator and synthesizer networks have identical architectures, including 
        the same number and sizes of hidden layers.

        The forward pass integrates the modulator's influence into the synthesizer network 
        by adding the modulator's output to the synthesizer's values at each layer, 
        followed by the application of activation functions.

        Args:
            latent_array (jax.Array): The input to the modulator network, provided as a flat 
                array that is reshaped internally for processing.
            coord_matrix (jax.Array): The input to the synthesizer network, typically 
                representing task-specific coordinates or features.
            modulator_nn (MLP): The modulator neural network, which generates modulating 
                parameters for the synthesizer network.
            synthesizer_nn (MLP): The synthesizer neural network, which computes the primary 
                task-specific outputs.

        Returns:
            jax.Array: The output of the synthesizer network after applying the "all-to-all" 
            coupling mechanism, integrating the influence of the modulator network.

        Process:
            1. Reshape `latent_array` to prepare it as input to the modulator network.
            2. Perform layer-by-layer computations:
            - Compute layer for the modulator network.
            - Compute layer for the synthesizer network.
            - Add the modulator's output to the synthesizer's layer values.
            - Apply activation functions to both modulator and synthesizer outputs.
            3. At the final layer, compute only the synthesizer's output using the 
            modulated activations.

        Notes:
            - Skip connections are supported in both networks if configured. They are 
            applied periodically based on the specified frequency in the network's 
            settings.
            - Activation functions are applied layer-wise with scaling gains, as specified 
            in the respective networks.

        Raises:
            ValueError: If there is a mismatch in the architectures of the modulator 
            and synthesizer networks when using "all-to-all" coupling.
        """    

        x_modul = latent_array.reshape(-1,1).T
        x_synth = coord_matrix

        if modulator_nn.skip_connections_settings["active"]:
            x_modul_init = x_modul.copy()

        if synthesizer_nn.skip_connections_settings["active"]:
            x_synth_init = x_synth.copy()        

        layer_num = 0
        for i in range(len(modulator_nn.nn_params)):
            (w_modul, b_modul) = modulator_nn.nn_params[i]
            (w_synth, b_synth) = synthesizer_nn.nn_params[i]
            
            # first compute x_modul
            if layer_num>0 and modulator_nn.skip_connections_settings["active"] and layer_num%modulator_nn.skip_connections_settings["frequency"]==0:
                x_modul = modulator_nn.ComputeXSkip(w_modul,x_modul,x_modul_init,b_modul)
            else:
                x_modul = modulator_nn.ComputeX(w_modul,x_modul,b_modul)

            # now compute x_synth
            if layer_num>0 and synthesizer_nn.skip_connections_settings["active"] and layer_num%synthesizer_nn.skip_connections_settings["frequency"]==0:
                x_synth = synthesizer_nn.ComputeXSkip(w_synth,x_synth,x_synth_init,b_synth)
            else:
                x_synth = synthesizer_nn.ComputeX(w_synth,x_synth,b_synth)

            # add x_modul
            x_synth = jax.vmap(lambda row: row + x_modul.reshape(-1))(x_synth)

            # now apply modul activation
            x_modul = modulator_nn.act_func(modulator_nn.act_func_gain*x_modul)
            # now apply synth activation
            x_synth = synthesizer_nn.act_func(synthesizer_nn.act_func_gain*x_synth)
            # update layer index
            layer_num += 1

        final_w_synth, final_b_synth = synthesizer_nn.nn_params[-1]

        if layer_num>0 and synthesizer_nn.skip_connections_settings["active"] and layer_num%synthesizer_nn.skip_connections_settings["frequency"]==0:
            return synthesizer_nn.ComputeXSkip(final_w_synth,x_synth,x_synth_init,final_b_synth)   
        else:
            return synthesizer_nn.ComputeX(final_w_synth,x_synth,final_b_synth)   

    def last_to_all_fw(self,latent_array:jax.Array,coord_matrix:jax.Array,
                            modulator_nn:MLP,synthesizer_nn:MLP):
        """
        Implements the "last-to-all" forward propagation coupling mechanism.

        In this coupling mode, the final output of the modulator network is added 
        to the activations of every layer of the synthesizer network during forward 
        propagation. This allows the modulator network's influence to span all layers 
        of the synthesizer network.

        Args:
            latent_array (jax.Array): The input to the modulator network, provided as 
                a flat array that is reshaped internally for processing.
            coord_matrix (jax.Array): The input to the synthesizer network, typically 
                representing task-specific coordinates or features.
            modulator_nn (MLP): The modulator neural network, which computes the 
                modulating output.
            synthesizer_nn (MLP): The synthesizer neural network, which computes 
                the primary task-specific outputs.

        Returns:
            jax.Array: The output of the synthesizer network after applying the 
            "last-to-all" coupling mechanism.

        Process:
            1. Forward propagate `latent_array` through the modulator network to produce 
            its final output (`x_modul`).
            2. Initialize the input for the synthesizer network (`x_synth`) using 
            `coord_matrix`.
            3. For each layer of the synthesizer network:
            - Compute the layer output of the synthesizer network (`x_synth`).
            - Add the corresponding part of `x_modul` to the layer biases.
            - Apply the synthesizer's activation function to the updated outputs.
            4. At the final layer, compute the output of the synthesizer network 
            without adding modulator outputs, unless configured otherwise.

        Notes:
            - Skip connections are supported in the synthesizer network if configured, 
            and are applied periodically based on the specified frequency.
            - The coupling is done by splitting `x_modul` into chunks that match the 
            biases of each synthesizer layer.

        Raises:
            ValueError: If the size of `x_modul` does not match the total biases in 
            the synthesizer network when using "last-to-all" coupling.
        """

        # first modulator fw
        x_modul = modulator_nn(latent_array.reshape(-1,1).T).flatten()
        x_synth = coord_matrix

        if synthesizer_nn.skip_connections_settings["active"]:
            x_synth_init = x_synth.copy()  

        # now synthesizer fw with shift modulation coupling
        x_modul_itr = 0
        layer_num = 0
        for i in range(len(synthesizer_nn.nn_params)-1):
            (w_synth, b_synth) = synthesizer_nn.nn_params[i]
            # get num of the hidden layer biases
            num_hidden_biases = b_synth.shape[0]
            # now compute x_synth
            if layer_num>0 and synthesizer_nn.skip_connections_settings["active"] and layer_num%synthesizer_nn.skip_connections_settings["frequency"]==0:
                x_synth = synthesizer_nn.ComputeXSkip(w_synth,x_synth,x_synth_init,b_synth)
            else:
                x_synth = synthesizer_nn.ComputeX(w_synth,x_synth,b_synth)
            # add x_modul
            this_layer_x_modul = x_modul[x_modul_itr:x_modul_itr+num_hidden_biases]
            x_synth = jax.vmap(lambda row: row + this_layer_x_modul)(x_synth)
            # now apply synth activation
            x_synth = synthesizer_nn.act_func(synthesizer_nn.act_func_gain*x_synth)
            # update x_modul_itr
            x_modul_itr += num_hidden_biases
            # update layer index
            layer_num += 1

        # now apply last linear layer
        final_w_synth, final_b_synth = synthesizer_nn.nn_params[-1]
        if layer_num>0 and synthesizer_nn.skip_connections_settings["active"] and layer_num%synthesizer_nn.skip_connections_settings["frequency"]==0:
            return synthesizer_nn.ComputeXSkip(final_w_synth,x_synth,x_synth_init,final_b_synth)   
        else:
            return synthesizer_nn.ComputeX(final_w_synth,x_synth,final_b_synth) 
        
    def one_modulator_per_synthesizer_layer_fw(self,latent_array:jax.Array,coord_matrix:jax.Array,
                                               modulator_nns:list[MLP],synthesizer_nn:MLP):
        """
        Implements the "one-modulator-per-synthesizer-layer" forward propagation coupling mechanism.

        In this coupling mode, each layer of the synthesizer network is modulated by a 
        dedicated modulator network. This allows for fine-grained control and layer-specific 
        modulation, where each modulator network processes the `latent_array` independently 
        to compute the modulation for its corresponding synthesizer layer.

        Args:
            latent_array (jax.Array): The input to all modulator networks, provided as a flat 
                array that is reshaped internally for processing.
            coord_matrix (jax.Array): The input to the synthesizer network, typically representing 
                task-specific coordinates or features.
            modulator_nns (list[MLP]): A list of modulator networks, with each modulator corresponding 
                to a specific layer of the synthesizer network.
            synthesizer_nn (MLP): The synthesizer neural network, which computes the primary 
                task-specific outputs.

        Returns:
            jax.Array: The output of the synthesizer network after applying the 
            "one-modulator-per-synthesizer-layer" coupling mechanism.

        Process:
            1. Initialize the input for the synthesizer network (`x_synth`) using `coord_matrix`.
            2. For each layer of the synthesizer network:
            - Compute the layer output of the synthesizer network (`x_synth`).
            - Compute the modulation for the layer using the corresponding modulator network.
            - Add the modulator's output to the layer's biases in `x_synth`.
            - Apply the synthesizer's activation function to the updated layer output.
            3. At the final layer, compute the output of the synthesizer network 
            without modulation from the modulator networks.
            4. If skip connections are enabled in the synthesizer network, apply them 
            based on the configured frequency.

        Notes:
            - Skip connections are supported in the synthesizer network and are applied 
            periodically based on the specified frequency.
            - The number of modulator networks (`modulator_nns`) must match the number 
            of layers in the synthesizer network, excluding the final layer.

        Raises:
            ValueError: If the number of modulator networks does not match the number 
            of layers in the synthesizer network.
        """
        x_synth = coord_matrix
        if synthesizer_nn.skip_connections_settings["active"]:
            x_synth_init = x_synth.copy()  

        layer_num = 0
        for i in range(len(synthesizer_nn.nn_params)-1):
            (w_synth, b_synth) = synthesizer_nn.nn_params[i]
            # now compute x_synth
            if layer_num>0 and synthesizer_nn.skip_connections_settings["active"] and layer_num%synthesizer_nn.skip_connections_settings["frequency"]==0:
                x_synth = synthesizer_nn.ComputeXSkip(w_synth,x_synth,x_synth_init,b_synth)
            else:
                x_synth = synthesizer_nn.ComputeX(w_synth,x_synth,b_synth)
            # compute x_modul
            this_layer_x_modul = modulator_nns[i](latent_array.reshape(-1,1).T).flatten()
            # now add to x_synth
            x_synth = jax.vmap(lambda row: row + this_layer_x_modul)(x_synth)
            # now apply synth activation
            x_synth = synthesizer_nn.act_func(synthesizer_nn.act_func_gain*x_synth)
            # update layer index
            layer_num += 1

        # now apply last linear layer
        final_w_synth, final_b_synth = synthesizer_nn.nn_params[-1]
        if layer_num>0 and synthesizer_nn.skip_connections_settings["active"] and layer_num%synthesizer_nn.skip_connections_settings["frequency"]==0:
            return synthesizer_nn.ComputeXSkip(final_w_synth,x_synth,x_synth_init,final_b_synth)   
        else:
            return synthesizer_nn.ComputeX(final_w_synth,x_synth,final_b_synth) 
    
    def __call__(self, latent_array: jax.Array,coord_matrix: jax.Array):
        if self.coupling_settings["modulator_to_synthesizer_coupling_mode"] == "one_modulator_per_synthesizer_layer":
            return jax.vmap(self.fw_func,in_axes=(0, None, None, None))(latent_array,coord_matrix,self.modulator_nns,self.synthesizer_nn)
        else:
            return jax.vmap(self.fw_func,in_axes=(0, None, None, None))(latent_array,coord_matrix,self.modulator_nn,self.synthesizer_nn)
