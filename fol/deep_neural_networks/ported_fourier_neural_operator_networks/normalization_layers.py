"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: Dec, 2025
 License: FOL/LICENSE
"""

from __future__ import annotations
from typing import Optional

import jax.numpy as jnp
from flax import nnx
import jax

# JAX/Flax NNX implementation of normalization_layers.
#
# Ported from the original PyTorch implementation:
#   Repository: https://github.com/neuraloperator/neuraloperator
#   File: neuralop/layers/normalization_layers.py
#   Commit: 14c0f7320dc7c94e907a16fd276248df2d71407c (2025-11-14)
#   URL:
#     https://github.com/neuraloperator/neuraloperator/blob/14c0f7320dc7c94e907a16fd276248df2d71407c/neuralop/layers/normalization_layers.py
#
# Original code copyright (c) 2023 NeuralOperator developers
# Licensed under the MIT License.
#
# Note:
#   The PyTorch implementation operates in NCHW (channels-first) format,
#   while JAX/Flax NNX uses NHWC (channels-last). This port includes
#   careful transformations between channel orders to preserve behavior.

class AdaIN(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        in_channels: int,
        mlp: Optional[nnx.Module] = None,
        eps: float = 1e-5,
        *,
        rngs: nnx.Rngs,
    ):
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.eps = eps

        class MLP(nnx.Module):
            def __init__(self, embed_dim: int, in_channels: int, rngs: nnx.Rngs):
                # Linear(embed_dim → 512)
                self.fc1 = nnx.Linear(embed_dim, 512, rngs=rngs)

                # Linear(512 → 2 * in_channels)
                self.fc2 = nnx.Linear(512, 2 * in_channels, rngs=rngs)

            def __call__(self, x):
                x = self.fc1(x)
                x = jax.nn.gelu(x)    # GELU activation, same as PyTorch nn.GELU
                x = self.fc2(x)
                return x              # shape (2 * in_channels,)
            
        if mlp is None:
            mlp = MLP(embed_dim,in_channels,rngs)
        self.mlp = mlp

        self.embedding = None        

    def set_embedding(self, emb: jnp.ndarray):
        emb = jnp.asarray(emb).reshape((self.embed_dim,))
        self.embedding = nnx.data(emb)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:

        """Apply adaptive instance normalization to the input tensor."""
        assert self.embedding is not None, "AdaIN: update embeddding before running forward"

        weight, bias = jnp.split(self.mlp(self.embedding), 2, axis=-1)

        spatial_axes = tuple(range(1, x.ndim - 1))

        mean = jnp.mean(x, axis=spatial_axes, keepdims=True)
        var  = jnp.var(x,  axis=spatial_axes, keepdims=True)

        x_hat = (x - mean) / jnp.sqrt(var + self.eps)

        # affine transform
        broadcast_shape = (1,) * (x.ndim - 1) + (self.in_channels,)
        gamma = weight.reshape(broadcast_shape)
        beta  = bias.reshape(broadcast_shape)

        return x_hat * gamma + beta


class InstanceNorm(nnx.Module):
    def __init__(
        self,
        num_features: int,
        *,
        rngs: nnx.Rngs,
        **kwargs,
    ):
        self.num_features = num_features
        self.kwargs = kwargs

        self.eps: float = kwargs.pop("eps", 1e-5)
        affine: bool = kwargs.pop("affine", True)

        weight_init = kwargs.pop("weight", None)
        bias_init = kwargs.pop("bias", None)

        if affine:
            if weight_init is not None:
                w_val = jnp.asarray(weight_init, dtype=jnp.float32)
            else:
                w_val = jnp.ones((num_features,), dtype=jnp.float32)

            if bias_init is not None:
                b_val = jnp.asarray(bias_init, dtype=jnp.float32)
            else:
                b_val = jnp.zeros((num_features,), dtype=jnp.float32)

            self.weight = nnx.Param(w_val)
            self.bias = nnx.Param(b_val)
        else:
            self.weight = None
            self.bias = None

        # Dummy use of rngs (to follow NNX API convention)
        self._rngs = rngs

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        size = x.shape
        assert x.ndim >= 2, f"Expected at least 2D input, got {x.ndim}D"
        assert x.shape[-1] == self.num_features, (
            f"Expected last dimension (channels) = {self.num_features}, "
            f"got {x.shape[-1]}"
        )

        spatial_axes = tuple(range(1, x.ndim - 1))

        mean = jnp.mean(x, axis=spatial_axes, keepdims=True)
        var = jnp.var(x, axis=spatial_axes, keepdims=True)

        x_hat = (x - mean) / jnp.sqrt(var + self.eps)

        if self.weight is not None and self.bias is not None:
            broadcast_shape = (1,) * (x.ndim - 1) + (self.num_features,)
            gamma = jnp.reshape(self.weight.value, broadcast_shape)
            beta = jnp.reshape(self.bias.value, broadcast_shape)
            x_hat = x_hat * gamma + beta

        assert x_hat.shape == size
        return x_hat

class BatchNorm(nnx.Module):
    """
    Dimension-agnostic batch normalization layer for neural operators (Flax NNX, channel-last).

    Differences from the PyTorch version:
    - Expects channel-last inputs: (N, *spatial_dims, C)
      instead of channel-first (N, C, *spatial_dims).
    - JAX/Flax do not have the 4D+ BatchNorm limitation, so we don't need to
      flatten spatial dims for n_dim > 3.

    Parameters
    ----------
    n_dim : int
        Spatial dimension of input data (1 for 1D, 2 for 2D, 3 for 3D, ...).
        We expect x.ndim == n_dim + 2  (batch + spatial + channel).
    num_features : int
        Number of channels (C) in the last dimension.
    eps : float
        Numerical stability epsilon (maps to `epsilon`).
    momentum : float
        Momentum for running statistics.
    affine : bool
        If True, use learnable scale and bias.
    track_running_stats : bool
        Kept only for API parity with PyTorch; not used.
    """

    def __init__(
        self,
        n_dim: int,
        num_features: int,
        *,
        rngs: nnx.Rngs,
        **kwargs
    ):
        self.n_dim = n_dim
        self.num_features = num_features

        eps = kwargs.pop("eps", 1e-5)
        momentum = kwargs.pop("momentum", 0.1)
        affine = kwargs.pop("affine", True)
        kwargs.pop("track_running_stats", None)

        self.norm = nnx.BatchNorm(
            num_features=num_features,
            axis=-1,
            epsilon=eps,
            momentum=momentum,
            use_scale=affine,
            use_bias=affine,
            use_running_average=False,
            rngs=rngs,
            **kwargs,      # passthrough BN kwargs
        )

    def __call__(self, x):
        assert x.shape[-1] == self.num_features
        assert x.ndim == self.n_dim + 2
        return self.norm(x)
