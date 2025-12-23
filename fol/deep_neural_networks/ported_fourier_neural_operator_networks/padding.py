"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: Dec, 2025
 License: FOL/LICENSE
"""

from __future__ import annotations

from typing import List, Union, Dict, Tuple, Any, Optional
import jax.numpy as jnp
from flax import nnx

Number = Union[float, int]

# JAX/Flax NNX implementation of padding.
# Ported from the original PyTorch implementation:
# https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/padding.py
#
# Original code copyright (c) 2023 NeuralOperator developers
# Licensed under the MIT License.
#
# Note:
#   The PyTorch implementation operates in NCHW (channels-first) format,
#   while JAX/Flax NNX uses NHWC (channels-last). This port includes
#   careful transformations between channel orders to preserve the
#   original module's behavior while conforming to Flax/JAX conventions.
#
# This file contains a reimplementation and may include modifications
# as required by the channel-layout differences.

def validate_scaling_factor(
    scaling_factor: Union[None, Number, List[Number], List[List[Number]]],
    n_dim: int,
    n_layers: Optional[int] = None,
) -> Union[None, List[float], List[List[float]]]:
    """
    JAX version of the PyTorch validate_scaling_factor.

    Parameters
    ----------
    scaling_factor : None OR float OR list[float] OR list[list[float]]
    n_dim : int
    n_layers : int or None; defaults to None
        If None, return a single list (rather than a list of lists)
        with `factor` repeated `dim` times.
    """
    if scaling_factor is None:
        return None

    if isinstance(scaling_factor, (float, int)):
        if n_layers is None:
            return [float(scaling_factor)] * n_dim
        return [[float(scaling_factor)] * n_dim] * n_layers

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all(isinstance(s, (float, int)) for s in scaling_factor)
    ):
        if n_layers is None and len(scaling_factor) == n_dim:
            # dim-wise scaling
            return [float(s) for s in scaling_factor]
        # layer-wise scaling factors; expand each scalar to dim
        return [[float(s)] * n_dim for s in scaling_factor]

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all(isinstance(s, list) for s in scaling_factor)
    ):
        s_sub_pass = True
        for s in scaling_factor:
            if all(isinstance(s_sub, (int, float)) for s_sub in s):
                pass
            else:
                s_sub_pass = False
        if s_sub_pass:
            return scaling_factor  # already list[list[float]]

    return None

class DomainPadding(nnx.Module):
    """
    Applies domain padding scaled automatically to the input's resolution.

    JAX / NNX version, assuming NHWC layout:

        input shape: (batch, d1, ..., dN, channels)

    Parameters
    ----------
    domain_padding : float or list
        Typically, between zero and one, percentage of padding to use.
        If a list, it must match the number of spatial dims (d1..dN).
    resolution_scaling_factor : int, list, or None
        Resolution scaling factor, by default 1.
    """

    def __init__(
        self,
        domain_padding: Union[float, list],
        resolution_scaling_factor: Union[int, List[int], None] = 1,
    ):
        super().__init__()
        self.domain_padding = domain_padding
        if resolution_scaling_factor is None:
            resolution_scaling_factor = 1
        self.resolution_scaling_factor: Union[int, List[int], None] = (
            resolution_scaling_factor
        )

        # dict[str(resolution)] = pad_width list for jnp.pad
        self._padding: Dict[str, List[Tuple[int, int]]] = {}

        # dict[str(output_shape_scaled_list)] = indices_to_unpad (tuple of slices)
        self._unpad_indices: Dict[str, Tuple[Any, ...]] = {}

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Default forward: pad the input."""
        return self.pad(x)

    def pad(self, x: jnp.ndarray, verbose: bool = False) -> jnp.ndarray:
        """
        Take an input and pad it by the desired fraction.

        The amount of padding will be automatically scaled with the resolution.

        x shape (NHWC): (batch, d1, ..., dN, channels)
        """
        spatial_dims = x.shape[1:-1]  # drop batch & channels
        n_spatial = len(spatial_dims)
        resolution_key = f"{tuple(spatial_dims)}"

        # Normalize domain_padding to list
        if isinstance(self.domain_padding, (float, int)):
            domain_padding_list = [float(self.domain_padding)] * n_spatial
        else:
            domain_padding_list = [float(p) for p in self.domain_padding]

        assert len(domain_padding_list) == n_spatial, (
            "domain_padding length must match the number of spatial/time dimensions "
            "(excluding batch, channels)"
        )

        # Resolution scaling factor handling (mimics PyTorch behavior)
        resolution_scaling_factor = self.resolution_scaling_factor
        if not isinstance(self.resolution_scaling_factor, list):
            # n_layers=None as in original DomainPadding
            resolution_scaling_factor = validate_scaling_factor(
                self.resolution_scaling_factor, n_spatial, n_layers=None
            )

        # If None (weird case), just default to 1.0 per dim for safety
        if resolution_scaling_factor is None:
            resolution_scaling_factor = [1.0] * n_spatial

        # Use cached padding if available
        if resolution_key in self._padding:
            pad_width = self._padding[resolution_key]
            return jnp.pad(x, pad_width, mode="constant")

        # --- Compute base padding per spatial dim (like PyTorch version) ---
        base_padding = [
            round(p * r) for (p, r) in zip(domain_padding_list, spatial_dims)
        ]  # length n_spatial

        if verbose:
            print(
                f"Padding inputs of resolution={spatial_dims} with "
                f"padding={base_padding}, symmetric"
            )

        # Scale for unpad index key (same logic as original)
        output_pad = base_padding
        output_pad = [
            round(i * j) for (i, j) in zip(resolution_scaling_factor, output_pad)
        ]

        # Build unpad slices for spatial dims
        unpad_slices = []
        for p in output_pad:
            if p == 0:
                padding_end = None
                padding_start = None
            else:
                padding_end = p
                padding_start = -p
            unpad_slices.append(slice(padding_end, padding_start, None))

        # NHWC indices: (batch, *spatial, channels)
        unpad_indices = (slice(None), *unpad_slices, slice(None))

        # Build pad_width for jnp.pad:
        # axis 0: batch        -> (0, 0)
        # axes 1..n_spatial    -> (p, p)
        # last axis: channels  -> (0, 0)
        pad_width: List[Tuple[int, int]] = [(0, 0)]
        for p in base_padding:
            pad_width.append((p, p))
        pad_width.append((0, 0))

        # Cache
        self._padding[resolution_key] = pad_width

        # Apply pad
        padded = jnp.pad(x, pad_width, mode="constant")

        # Compute padded spatial shape
        padded_spatial = padded.shape[1:-1]

        # Adjust for scaling factor (same as PyTorch)
        output_shape_scaled = [
            round(i * j) for (i, j) in zip(padded_spatial, resolution_scaling_factor)
        ]

        # Store unpad indices keyed by the scaled spatial shape
        unpad_key = f"{[i for i in output_shape_scaled]}"
        self._unpad_indices[unpad_key] = unpad_indices

        return padded

    def unpad(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Remove the padding from padded inputs.

        x: NHWC (batch, d1, ..., dN, channels)
        """
        spatial_dims = x.shape[1:-1]
        unpad_key = f"{[int(s) for s in spatial_dims]}"
        unpad_indices = self._unpad_indices[unpad_key]
        return x[unpad_indices]
