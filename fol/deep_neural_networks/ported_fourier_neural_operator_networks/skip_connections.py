"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: Dec, 2025
 License: FOL/LICENSE
"""

from __future__ import annotations
from typing import Optional, Literal

import jax.numpy as jnp
from flax import nnx

SkipType = Literal["soft-gating", "linear", "identity"]

# JAX/Flax NNX implementation of skip_connections.
#
# Ported from the original PyTorch implementation:
#   Repository: https://github.com/neuraloperator/neuraloperator
#   File: neuralop/layers/skip_connections.py
#   Commit: 14c0f7320dc7c94e907a16fd276248df2d71407c (2025-11-14)
#   URL:
#     https://github.com/neuraloperator/neuraloperator/blob/14c0f7320dc7c94e907a16fd276248df2d71407c/neuralop/layers/skip_connections.py
#
# Original code copyright (c) 2023 NeuralOperator developers
# Licensed under the MIT License.
#
# Note:
#   The PyTorch implementation operates in NCHW (channels-first) format,
#   while JAX/Flax NNX uses NHWC (channels-last). This port includes
#   careful transformations between channel orders to preserve behavior.

def skip_connection(
    in_features: int,
    out_features: int,
    n_dim: int = 2,
    bias: bool = False,
    skip_type: SkipType = "soft-gating",
    *,
    rngs: nnx.Rngs,
):
    """A wrapper for several types of skip connections.
    Returns an nn.Module skip connections, one of  {'identity', 'linear', soft-gating'}

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    n_dim : int, optional
        Dimensionality of the input (excluding batch-size and channels).
        n_dim=2 corresponds to having Module2D, by default 2
    bias : bool, optional
        Whether to use a bias, by default False
    skip_type : {'identity', 'linear', 'soft-gating'}, optional
        Kind of skip connection to use. Options: 'identity', 'linear', 'soft-gating', by default "soft-gating"

    Returns
    -------
    nn.Module
        module that takes in x and returns skip(x)
    """
    if skip_type.lower() == "soft-gating":
        return SoftGating(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            n_dim=n_dim,
        )
    elif skip_type.lower() == "linear":
        return Flattened1dConv(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            bias=bias,
            rngs=rngs
        )
    elif skip_type.lower() == "identity":
        return Identity()
    else:
        raise ValueError(
            f"Got skip-connection type={skip_type}, expected one of"
            f" {'soft-gating', 'linear', 'id'}."
        )


class Identity(nnx.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

class SoftGating(nnx.Module):
    """
    Channel-last version of SoftGating.

    PyTorch version:
        x: (N, C, H, W, ...)
        weight: (1, C, 1, 1, ...)
        y = weight * x (+ bias)

    NNX version:
        x: (N, ..., C)
        weight: (C,)
        y = x * weight (+ bias)  (broadcast over last axis)
    """

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        n_dim: int = 2,
        bias: bool = False,
    ):
        if out_features is not None and in_features != out_features:
            raise ValueError(
                f"Got in_features={in_features} and out_features={out_features}, "
                "but these two must be the same for soft-gating"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.n_dim = n_dim

        # Channel weights (one scalar per channel)
        self.weight = nnx.Param(jnp.ones((in_features,), dtype=jnp.float32))
        self.bias = (
            nnx.Param(jnp.ones((in_features,), dtype=jnp.float32)) if bias else None
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (N, *spatial_dims, C)
        """
        assert x.shape[-1] == self.in_features, (
            f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
        )

        y = x * self.weight  # broadcasts over last axis
        if self.bias is not None:
            y = y + self.bias
        return y


class Flattened1dConv(nnx.Module):
    """
    Channel-last version of Flattened1dConv.

    PyTorch version:
        x: (B, C, x1, ..., xn)
        - flatten spatial dims -> (B, C, L)
        - Conv1d over L -> (B, C_out, L_out)
        - reshape -> (B, C_out, x1, ..., xn)  (assumes L_out == L)

    NNX version:
        x: (B, x1, ..., xn, C)
        - flatten spatial dims -> (B, L, C)
        - Conv1d (channel-last) over L -> (B, L_out, C_out)
        - reshape -> (B, x1, ..., xn, C_out)  (assumes L_out == L)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # NNX Conv: input (B, L, C_in), kernel (K, C_in, C_out), output (B, L_out, C_out)
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(kernel_size,),
            strides=(1,),
            padding="VALID",
            use_bias=bias,
            rngs=rngs,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (B, *spatial_dims, C_in)
        returns: (B, *spatial_dims, C_out)  (if kernel_size=1)
        """
        size = x.shape
        assert x.ndim >= 3, "Expected at least (B, spatial, C) tensor"
        assert size[-1] == self.in_channels, (
            f"Expected last dimension (channels)={self.in_channels}, "
            f"got {size[-1]}"
        )

        B = size[0]
        spatial = size[1:-1]       # (x1, ..., xn)
        C_in = size[-1]

        # Flatten spatial dims -> (B, L, C_in)
        x_flat = x.reshape(B, -1, C_in)  # L = prod(spatial)
        y_flat = self.conv(x_flat)       # (B, L_out, C_out)

        # To faithfully mimic the PyTorch behavior where they reshape
        # back to *size[2:], we require L_out == L.
        assert y_flat.shape[1] == x_flat.shape[1], (
            "Flattened length changed by Conv1d; to preserve original spatial shape, "
            "use kernel_size=1 (and stride=1, padding='VALID')."
        )

        # Reshape back to (B, *spatial_dims, C_out)
        y = y_flat.reshape(B, *spatial, self.out_channels)
        return y
