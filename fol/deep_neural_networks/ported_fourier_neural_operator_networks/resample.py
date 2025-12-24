"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: Dec, 2025
 License: FOL/LICENSE
"""

import itertools
import jax.numpy as jnp
import jax.image as jimage

# JAX/Flax NNX implementation of resample.
#
# Ported from the original PyTorch implementation:
#   Repository: https://github.com/neuraloperator/neuraloperator
#   File: neuralop/layers/resample.py
#   Commit: 14c0f7320dc7c94e907a16fd276248df2d71407c (2025-11-14)
#   URL:
#     https://github.com/neuraloperator/neuraloperator/blob/14c0f7320dc7c94e907a16fd276248df2d71407c/neuralop/layers/resample.py
#
# Original code copyright (c) 2023 NeuralOperator developers
# Licensed under the MIT License.
#
# Note:
#   The PyTorch implementation operates in NCHW (channels-first) format,
#   while JAX/Flax NNX uses NHWC (channels-last). This port includes
#   careful transformations between channel orders to preserve behavior.

def resample(x, res_scale, axis, output_shape=None):
    """Generic n-dimensional interpolation (JAX/Flax, channel-last).

    Layout
    ------
    x : (batch, d1, ..., dN, channels)

    Parameters
    ----------
    x : jax.Array
        Input activation of size (B, d1, ..., dN, C).
    res_scale : int, float, or tuple
        Scaling factor(s) along each of the dimensions in `axis`.
        If scalar, isotropic scaling is performed.
    axis : int, list, or tuple
        Axis/axes along which interpolation will be performed.
        For this channel-last version, we assume spatial dims are 1..(x.ndim-2).
        For the FFT (N>=3) branch, `axis` must cover all spatial dims.
    output_shape : None or tuple[int]
        If provided, directly sets the target spatial shape instead of
        computing it from `res_scale`.
    """

    nd = x.ndim
    if nd < 3:
        raise ValueError("Input must be at least 3D: (B, spatial..., C)")

    # Spatial axes for channel-last
    spatial_axes = list(range(1, nd - 1))  # d1..dN

    # Normalize axis to list
    if axis is None:
        axis = spatial_axes
    elif isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis)

    # Normalize res_scale to list
    if isinstance(res_scale, (float, int)):
        res_scale = [res_scale] * len(axis)
    else:
        assert len(res_scale) == len(axis), "length of res_scale and axis are not same"

    # Old spatial sizes along selected axes
    old_size = tuple(x.shape[a] for a in axis)

    if output_shape is None:
        new_size = tuple(int(round(s * r)) for (s, r) in zip(old_size, res_scale))
    else:
        new_size = tuple(output_shape)

    spatial_ndim = len(spatial_axes)

    # --- 1D case: (B, L, C) with one spatial dimension ---
    if len(axis) == 1 and spatial_ndim == 1:
        a = axis[0]
        target_shape = list(x.shape)
        target_shape[a] = new_size[0]
        return jimage.resize(x, shape=tuple(target_shape), method="linear")

    # --- 2D case: (B, H, W, C) with two spatial dimensions ---
    if len(axis) == 2 and spatial_ndim == 2:
        target_shape = list(x.shape)
        for ax, ns in zip(axis, new_size):
            target_shape[ax] = ns
        # "cubic" ~ bicubic (not exactly PyTorch's, but similar)
        return jimage.resize(x, shape=tuple(target_shape), method="cubic")
    
    # --- 3D+ case: spectral interpolation using FFT ---
    # Ensure axis is a sequence for jnp.fft.rfftn
    if isinstance(axis, int):
        axes = (axis,)
    else:
        axes = tuple(axis)

    X = jnp.fft.rfftn(x, axes=axes, norm="forward")

    new_fft_size = list(new_size)
    # Redundant last coefficient for rfftn: N → N//2 + 1
    new_fft_size[-1] = new_fft_size[-1] // 2 + 1

    # Size to copy (min of target and current FFT size)
    X_spatial_fft_shape = X.shape[-len(axes):]
    new_fft_size_c = [min(i, j) for (i, j) in zip(new_fft_size, X_spatial_fft_shape)]

    # Allocate output FFT array
    out_fft_shape = [x.shape[0], x.shape[1], *new_fft_size]
    out_fft = jnp.zeros(out_fft_shape, dtype=jnp.complex64)

    # Build index boundaries like original (copy low and high modes)
    # mode_indexing: for each spatial dim except last (which is rfft),
    # split around Nyquist. Last dim: simple slice from 0 to new_fft_size_c[-1].
    mode_indexing = [
        ((None, m // 2), (-m // 2, None)) for m in new_fft_size_c[:-1]
    ] + [((None, new_fft_size_c[-1]),)]

    # Iterate over all combinations of low/high frequency slices
    for boundaries in itertools.product(*mode_indexing):
        idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]
        idx_tuple = tuple(idx_tuple)
        out_fft = out_fft.at[idx_tuple].set(X[idx_tuple])

    y = jnp.fft.irfftn(out_fft, s=new_size, axes=axes, norm="forward")
    return y