"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: June, 2025
 License: FOL/LICENSE
"""
import copy
from flax import nnx
from .nns import MLP
import jax
from fol.tools.usefull_functions import *
from fol.tools.decoration_functions import *
from jax.nn import relu,sigmoid,swish,tanh,leaky_relu,elu

class DeepONet(nnx.Module):

    @print_with_timestamp_and_execution_time
    def __init__(self,name:str,
                      branch_nn:MLP,
                      trunk_nn:MLP,
                      output_dimension:int,
                      output_scale_factor:float=1.0,
                      activation_function_name:str=None,
                      use_bias:bool=True):
        self.name = name
        self.branch_nn = branch_nn
        self.trunk_nn = trunk_nn
        self.output_dimension = output_dimension
        self.output_scale_factor = output_scale_factor
        self.activation_function_name = activation_function_name
        self.use_bias = use_bias

        if self.trunk_nn.out_features != self.branch_nn.out_features:
            fol_error(f"trunk out_features:{self.trunk_nn.out_features} is not equal to branch out_features:{self.branch_nn.out_features}")

        self.dot_prod_vector_size = int(self.trunk_nn.out_features/self.output_dimension)

        if self.activation_function_name == None:
            self.act_func = lambda x: x
        else:
            self.act_func = globals()[self.activation_function_name]

        total_num_weights = self.branch_nn.total_num_weights + self.trunk_nn.total_num_weights
        total_num_biases = self.branch_nn.total_num_biases + self.trunk_nn.total_num_biases

        if self.use_bias:
            self.output_bias = nnx.Param(jnp.zeros(self.output_dimension))
            total_num_biases += self.output_dimension
        else:
            self.output_bias = nnx.Variable(jnp.zeros(self.output_dimension))

        fol_info(f"DeepONet network has {total_num_weights} weights and {total_num_biases} biases !")

    def GetName(self):
        """
        Retrieves the name of the hypernetwork module.

        Returns:
            str: The name of the hypernetwork module.
        """
        return self.name
    
    def __call__(self, batch_branch_input: jax.Array,trunk_input: jax.Array):

        def single_predict(branch_input,trunk_input):

            branch_output = self.branch_nn(branch_input)
            trunk_output = self.act_func(self.trunk_nn(trunk_input))

            batch_size = trunk_output.shape[0]
            reshaped_trunk = trunk_output.reshape(batch_size, self.output_dimension, self.dot_prod_vector_size)
            reshaped_branch = branch_output.reshape(self.output_dimension, self.dot_prod_vector_size)

            return self.output_scale_factor * (jnp.einsum('bod,od->bo', reshaped_trunk, reshaped_branch) + self.output_bias)

        return jax.vmap(single_predict,in_axes=(0, None))(batch_branch_input,trunk_input)
