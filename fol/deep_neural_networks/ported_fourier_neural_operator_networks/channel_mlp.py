"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: Dec, 2025
 License: FOL/LICENSE
"""

from __future__ import annotations

from typing import Optional, Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx

# JAX/Flax NNX implementation of ChannelMLP.
#
# Ported from the original PyTorch implementation:
#   Repository: https://github.com/neuraloperator/neuraloperator
#   File: neuralop/layers/channel_mlp.py
#   Commit: 14c0f7320dc7c94e907a16fd276248df2d71407c (2025-11-14)
#   URL:
#     https://github.com/neuraloperator/neuraloperator/blob/14c0f7320dc7c94e907a16fd276248df2d71407c/neuralop/layers/channel_mlp.py
#
# Original code copyright (c) 2023 NeuralOperator developers
# Licensed under the MIT License.
#
# Note:
#   The PyTorch implementation operates in NCHW (channels-first) format,
#   while JAX/Flax NNX uses NHWC (channels-last). This port includes
#   careful transformations between channel orders to preserve behavior.


class ChannelMLP(nnx.Module):
    """Multi-layer perceptron applied channel-wise across spatial dimensions.

    ChannelMLP applies a sequence of linear projections and nonlinearities to the
    channel dimension of input tensors, making it invariant to spatial resolution.
    This is particularly useful in neural operators where spatial dimensions may vary
    but channel-wise processing should remain consistent across all positions.

    Unlike the original PyTorch implementation—which uses 1D convolutions with
    kernel size 1 applied to (B, C, L) channel-first tensors—this JAX/Flax NNX
    version operates on channels-last tensors of shape (B, d1, ..., dn, C). In this
    layout, a standard Linear layer naturally performs the same per-channel mixing
    as a Conv1d with kernel size 1, since both are equivalent to applying an MLP
    independently at each spatial location.

    Using Linear layers in NNX offers several advantages:

    * It directly matches the channels-last convention of JAX/Flax.
    * It avoids unnecessary convolution machinery and is computationally simpler.
    * It more clearly expresses the intended abstraction: an MLP over channels,
    broadcast across all spatial positions.
    * It preserves numerical equivalence to the Conv1d(kernel=1) operation.

    Internally, the spatial dimensions are flattened, the MLP is applied over the
    channel axis, and the output is reshaped back to the original spatial form.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int, optional
        Number of output channels. If None, defaults to in_channels.
    hidden_channels : int, optional
        Number of hidden channels in intermediate layers. If None, defaults to in_channels.
    n_layers : int, optional
        Number of linear layers in the MLP, by default 2.
    n_dim : int, optional
        Spatial dimensionality (unused but kept for API compatibility), by default 2.
    non_linearity : callable, optional
        Activation function applied between layers, by default jax.nn.gelu.
    dropout : float, optional
        Dropout probability applied after each layer (except the last). If 0, no dropout is applied,
        by default 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        n_layers: int = 2,
        n_dim: int = 2,  # unused, kept for API compatibility
        non_linearity: Optional[Callable] = jax.nn.gelu,
        dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.non_linearity = non_linearity

        # Dropout layers (one per layer, like PyTorch)
        if dropout > 0.0:
            self.dropouts: Optional[nnx.List[nnx.Module]] = nnx.List([
                nnx.Dropout(rate=dropout, rngs=rngs) for _ in range(n_layers)
            ])
        else:
            self.dropouts = None

        # Build linear layers
        self.fcs = nnx.List([])
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                # Single layer: input -> output
                lin = nnx.Linear(
                    rngs=rngs,
                    in_features=self.in_channels,
                    out_features=self.out_channels,
                )
            elif i == 0:
                # First layer: input -> hidden
                lin = nnx.Linear(
                    rngs=rngs,
                    in_features=self.in_channels,
                    out_features=self.hidden_channels,
                )
            elif i == (n_layers - 1):
                # Last layer: hidden -> output
                lin = nnx.Linear(
                    rngs=rngs,
                    in_features=self.hidden_channels,
                    out_features=self.out_channels,
                )
            else:
                # Internal: hidden -> hidden
                lin = nnx.Linear(
                    rngs=rngs,
                    in_features=self.hidden_channels,
                    out_features=self.hidden_channels,
                )
            self.fcs.append(lin)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Parameters
        ----------
        x : jnp.ndarray
            Input tensor of shape (batch, d1, ..., dn, in_channels)
        train : bool
            Whether we are in training mode (affects dropout).
        """
        size = x.shape  # (B, d1, ..., dn, C_in)
        B = size[0]
        spatial_dims = size[1:-1]
        C_in = size[-1]

        assert (
            C_in == self.in_channels
        ), f"Expected last dim (channels) = {self.in_channels}, got {C_in}"

        # Flatten spatial dims: (B, d1, ..., dn, C) -> (B, L, C)
        x = x.reshape((B, -1, C_in))

        # Apply MLP across the last dimension (C), broadcasting over (B, L)
        for i, fc in enumerate(self.fcs):
            x = fc(x)  # Linear along last axis
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropouts is not None:
                x = self.dropouts[i](x)

        # Restore original spatial dims with new channels
        x = x.reshape((B, *spatial_dims, self.out_channels))
        return x

class LinearChannelMLP(nnx.Module):
    """
    Multi-layer perceptron (MLP) for channel processing using fully connected layers.

    This is a Flax NNX port of the corresponding PyTorch implementation. It is an
    alternative to a convolution-based ChannelMLP, using standard Linear layers.

    The network is defined by `layers = [in_channels, hidden1, ..., out_channels]`
    and applies:
      - a Linear transformation at every layer,
      - `non_linearity` after every layer except the last,
      - Dropout after every Linear layer *including the last* (to match the PyTorch code)
        when `dropout > 0`.

    Parameters
    ----------
    layers : Sequence[int]
        Architecture definition: [in_channels, hidden1, ..., out_channels].
        Must have at least 2 elements (input and output sizes).
    non_linearity : Callable[[jnp.ndarray], jnp.ndarray], optional
        Activation function applied after each Linear layer except the last.
        Defaults to `jax.nn.gelu`.
    dropout : float, optional
        Dropout probability. If > 0, dropout is applied after each Linear layer
        (including the last) to match the PyTorch implementation. If 0, no dropout
        is applied. Defaults to 0.0.
    rngs : nnx.Rngs
        Random number generators used for parameter initialization and dropout.
    """

    def __init__(
        self,
        layers: Sequence[int],
        non_linearity: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.gelu,
        dropout: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1, (
            "Error: trying to instantiate a LinearChannelMLP "
            "with only one linear layer."
        )

        self.non_linearity = non_linearity

        # Fully connected layers
        self.fcs = nnx.List([])
        for j in range(self.n_layers):
            self.fcs.append(
                nnx.Linear(
                    in_features=layers[j],
                    out_features=layers[j + 1],
                    rngs=rngs,
                )
            )

        # Dropout layers (one per linear layer) or None
        if dropout > 0.0:
            self.dropout: Optional[nnx.List[nnx.Dropout]] = nnx.List([
                nnx.Dropout(rate=dropout, rngs=rngs) for _ in range(self.n_layers)
            ])
        else:
            self.dropout = None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Apply the linear channel MLP.

        The input is assumed to be a 2D array where the last dimension corresponds
        to channels/features. This method preserves the leading dimension(s) by
        applying per-row Linear transformations.

        Dropout (if enabled) is applied after each Linear layer, including the last,
        matching the behavior of the original PyTorch implementation.

        Parameters
        ----------
        x : jnp.ndarray
            Input array of shape (batch, in_channels) or (batch * spatial, in_channels).

        Returns
        -------
        jnp.ndarray
            Output array of shape (batch, out_channels) or (batch * spatial, out_channels).
        """
        for i, fc in enumerate(self.fcs):
            x = fc(x)  # Linear transformation

            # Nonlinearity on all but last layer
            if i < self.n_layers - 1:
                x = self.non_linearity(x)

            # Dropout after each layer (including last), matching the PyTorch code
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x