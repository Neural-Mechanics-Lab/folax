"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: Dec, 2025
 License: FOL/LICENSE
"""

from __future__ import annotations

from typing import List, Sequence, Union, Optional

import jax.numpy as jnp
from flax import nnx

Number = Union[float, int]

# JAX/Flax NNX implementation of GridEmbeddingND.
# Ported from the original PyTorch implementation:
# https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/layers/embeddings.py
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

def regular_grid_nd(
    resolutions: List[int], grid_boundaries: List[List[int]] = [[0, 1]] * 2
):
    """regular_grid_nd generates a tensor of coordinate points that
    describe a bounded regular grid.

    Creates a dim x res_d1 x ... x res_dn stack of positional encodings A, where
    A[:,c1,c2,...] = [[d1,d2,...dn]] at coordinate (c1,c2,...cn) on a (res_d1, ...res_dn) grid.

    Parameters
    ----------
    resolutions : List[int]
        resolution of the output grid along each dimension
    grid_boundaries : List[List[int]], optional
        List of pairs [start, end] of the boundaries of the
        regular grid. Must correspond 1-to-1 with resolutions default [[0,1], [0,1]]

    Returns
    -------
    grid: tuple(Tensor)
    list of tensors describing positional encoding
    """
    assert len(resolutions) == len(
        grid_boundaries
    ), "Error: inputs must have same number of dimensions"
    dim = len(resolutions)

    meshgrid_inputs = list()
    for res, (start, stop) in zip(resolutions, grid_boundaries):
        meshgrid_inputs.append(jnp.linspace(start, stop, res + 1)[:-1])

    return jnp.meshgrid(*meshgrid_inputs, indexing="ij")

class GridEmbeddingND(nnx.Module):
    """GridEmbeddingND applies a simple positional embedding as a regular ND grid.
    
    It expects inputs of shape (batch, d_1, ..., d_n, channels)

    Parameters
    ----------
    in_channels : int
        number of channels in input
    dim : int
        dimensions of positional encoding to apply
    grid_boundaries : list, optional
        coordinate boundaries of input grid along each dim, by default [[0, 1], [0, 1]]
        """

    def __init__(
        self,
        in_channels: int,
        dim: int = 2,
        grid_boundaries: Optional[List[List[Number]]] = [[0, 1], [0, 1]],
    ):
        self.in_channels = in_channels
        self.dim = dim
        assert self.dim == len(
            grid_boundaries
        ), f"Error: expected grid_boundaries to be an iterable of length {self.dim}, received {grid_boundaries}"
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    @property
    def out_channels(self) -> int:
        return self.in_channels + self.dim

    def grid(self, spatial_dims: Sequence[int], dtype:jnp.dtype) -> jnp.ndarray:
        """grid generates ND grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : sizes of spatial resolution
        dtype : str
            dtype to encode data

        Returns
        -------
        jnp.ndarray
            output grids to concatenate
        """

        if self._grid is None or self._res != spatial_dims:
            grids_by_dim = regular_grid_nd(spatial_dims, grid_boundaries=self.grid_boundaries)
            grid = nnx.data(jnp.stack(grids_by_dim, axis=-1).astype(dtype))
            self._grid = grid
            self._res = spatial_dims

        return self._grid

    def __call__(self, data: jnp.ndarray, batched: bool = True) -> jnp.ndarray:
        """
        Params
        --------
        data: torch.Tensor
            assumes shape (batch (optional), x_1, x_2, ...x_n, channels)
        batched: bool
            whether data has a batch dim
        """

        if not batched and data.ndim == self.dim + 1:
            data = data[None, ...]  # add batch axis

        batch_size = data.shape[0]
        spatial_dims = data.shape[1:-1]
        dtype = data.dtype
        grids = self.grid(spatial_dims=data.shape[1:-1], dtype=dtype)
        grids = jnp.broadcast_to(grids, (batch_size, *spatial_dims, self.dim))
        out = jnp.concatenate([data, grids], axis=-1)
        return out