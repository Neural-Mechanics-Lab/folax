"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: Dec, 2025
 License: FOL/LICENSE
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Optional, Tuple, List, Union

# JAX/Flax NNX implementation of spectral_convolution.
#
# Ported from the original PyTorch implementation:
#   Repository: https://github.com/neuraloperator/neuraloperator
#   File: neuralop/layers/spectral_convolution.py
#   Commit: 14c0f7320dc7c94e907a16fd276248df2d71407c (2025-11-14)
#   URL:
#     https://github.com/neuraloperator/neuraloperator/blob/14c0f7320dc7c94e907a16fd276248df2d71407c/neuralop/layers/spectral_convolution.py
#
# Original code copyright (c) 2023 NeuralOperator developers
# Licensed under the MIT License.
#
# Note:
#   The PyTorch implementation operates in NCHW (channels-first) format,
#   while JAX/Flax NNX uses NHWC (channels-last). This port includes
#   careful transformations between channel orders to preserve behavior.

from .resample import resample

Number = Union[int, float]

einsum_symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def contract_dense(x, weight, separable=False):

    order = x.ndim  # e.g. 5 for [B, M1, M2, M3, Cin]

    # batch, spatial..., in_channels
    x_syms = list(einsum_symbols[:order])   # ['a','b','c','d','e']
    batch_sym = x_syms[0]
    spatial_syms = x_syms[1:-1]            # ['b','c','d']
    last_sym = x_syms[-1]                  # 'e'  (Cin, the contracted dim)

    if separable:
        # x:      a b c d e
        # weight: e b c d e  (Cin, spatial..., Cin)
        # out:    a b c d e  (no new channel dim, depthwise-style)
        weight_syms = [last_sym] + spatial_syms + [last_sym]
        out_syms = [batch_sym] + spatial_syms + [last_sym]
    else:
        # introduce a new symbol for output channels
        out_ch_sym = einsum_symbols[order]  # e.g. 'f'

        # weight: e b c d f  -> shape (Cin, *spatial, Cout)
        weight_syms = [last_sym] + spatial_syms + [out_ch_sym]

        # out: a b c d f  -> shape (B, *spatial, Cout)
        out_syms = [batch_sym] + spatial_syms + [out_ch_sym]

    eq = f'{"".join(x_syms)},{"".join(weight_syms)}->{"".join(out_syms)}'

    return jnp.einsum(eq, x, weight)

class SpectralConv(nnx.Module):
    """
    Flax NNX port of the PyTorch SpectralConv (dense-only version).

    Unsupported features:
        - factorization (CP, Tucker, TT)
        - factorized contraction
        - FactorizedTensor
        - mixed/half precision modes

    Uses NHWC layout internally: (B, d1, d2, ..., C)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Union[int, List[int]],
        complex_data: bool = False,
        max_n_modes: Optional[Union[int, List[int]]] = None,
        bias: bool = True,
        separable: bool = False,
        resolution_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision: str = "full",
        rank: float = 1.0,
        factorization: Optional[str] = None,
        implementation: str = "reconstructed",
        fixed_rank_modes: bool = False,
        decomposition_kwargs: Optional[dict] = None,
        init_std: Union[str, float] = "auto",
        fft_norm: str = "forward",
        *,
        rngs: nnx.Rngs
    ):

        super().__init__()

        # ----------------------------
        # Reject factorization modes
        # ----------------------------
        if factorization is not None and factorization.lower() != "dense":
            raise NotImplementedError(
                f"factorization={factorization} is not implemented in the JAX/NNX port "
                "(Option C: dense-only mode)."
            )

        if complex_data:
            raise NotImplementedError(
                "complex_data=True is not supported in the JAX/NNX implementation yet."
            )

        self.complex_data = False
        self.fft_norm = fft_norm

        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = list(n_modes)
        self.order = len(self.n_modes)

        if max_n_modes is None:
            max_n_modes = list(self.n_modes)
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.fno_block_precision = fno_block_precision

        if separable and in_channels != out_channels:
            raise ValueError(
                "separable=True requires in_channels == out_channels."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.separable = separable

        # ----------------------------
        # Weight init
        # ----------------------------
        if init_std == "auto":
            init_std = (2.0 / (in_channels + out_channels)) ** 0.5

        # Dense weight tensor:
        # shape = (in_channels, *max_n_modes, out_channels)
        weight_shape = (in_channels, *max_n_modes, out_channels)
        self.weight = nnx.Param(
            jax.random.normal(rngs(), weight_shape) * init_std
        )

        # Optional bias: (1, 1, ..., 1, out_channels)
        if bias:
            bias_shape = tuple([1] * self.order) + (out_channels,)
            self.bias = nnx.Param(
                jax.random.normal(rngs(), bias_shape) * init_std
            )
        else:
            self.bias = None

        # Resolution scaling factor
        if isinstance(resolution_scaling_factor, (float, int)):
            resolution_scaling_factor = [resolution_scaling_factor] * self.order
        self.resolution_scaling_factor = resolution_scaling_factor

    # ----------------------------
    # Mode setter
    # ----------------------------
    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, nm):
        if isinstance(nm, int):
            nm = [nm]
        nm = list(nm)
        if not self.complex_data:
            nm[-1] = nm[-1] // 2 + 1
        self._n_modes = nm

    # ----------------------------
    # Resolution scaling (same logic)
    # ----------------------------
    def transform_shape(self, spatial_dims):
        if self.resolution_scaling_factor is None:
            return spatial_dims
        return tuple([
            round(s * r) for s, r in zip(spatial_dims, self.resolution_scaling_factor)
        ])

    # ----------------------------
    # Dense contraction using einsum
    # ----------------------------
    def contract_dense(self, x_fft: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
        """
        x_fft: (B, d1, ..., dN, Cin)
        w:     (Cin, Cout, m1, ..., mN)  with modes matching d* after slicing

        returns: (B, d1, ..., dN, Cout)
        """
        order = self.order  # number of spatial dims

        # choose distinct single-letter symbols
        batch_sym = "b"
        in_sym = "i"
        out_sym = "o"
        # use c,d,e,f,... for spatial dims
        spatial_syms = list("cdefghijklmnopqrstuvwxyz")[:order]

        # build subscripts
        # x: b s1 s2 ... sN i
        x_sub = batch_sym + "".join(spatial_syms) + in_sym

        # w: i o s1 s2 ... sN
        w_sub = in_sym + out_sym + "".join(spatial_syms)

        # out: b s1 s2 ... sN o
        out_sub = batch_sym + "".join(spatial_syms) + out_sym

        eq = f"{x_sub},{w_sub}->{out_sub}"
        # e.g. for 2D: "bchi,ioch->bcho"

        return jnp.einsum(eq, x_fft, w)
    
    def transform(self, x, output_shape=None):
        in_shape = list(x.shape[1:-1])

        if self.resolution_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [round(s * r) for (s, r) in zip(in_shape, self.resolution_scaling_factor)]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(x, 1.0, list(range(2, x.ndim)), output_shape=out_shape)

    # ----------------------------
    # Forward pass
    # ----------------------------
    def __call__(self, x: jnp.ndarray, output_shape: Optional[Tuple[int]] = None):
    
        """
        NHWC input:
        x: (batch_size, d1, ..., dN, channels)
        """

        # 1. Extract shape components (NHWC)
        batchsize = x.shape[0]
        mode_sizes = list(x.shape[1:-1])  # [d1, ..., dN]
        channels = x.shape[-1]


        # 2. Compute fft_size (frequency-domain spatial sizes)
        fft_size = list(mode_sizes)
        if not self.complex_data:
            # Real FFT: last spatial dim becomes n//2 + 1
            fft_size[-1] = fft_size[-1] // 2 + 1

        fft_dims = tuple(range(1, x.ndim - 1))

        if self.fno_block_precision == "half":
            x = x.astype(jnp.float16)       

        if self.complex_data:    
            raise NotImplementedError(
                "complex_data=True is not supported in the JAX/NNX implementation yet."
            )
        else:
            x = jnp.fft.rfftn(x, axes=fft_dims, norm=self.fft_norm) 
            # When x is real in spatial domain, the last half of the last dim is redundant.
            # See :ref:`fft_shift_explanation` for discussion of the FFT shift.
            dims_to_fft_shift = fft_dims[:-1]    

        if self.order > 1:
            x = jnp.fft.fftshift(x, axes=dims_to_fft_shift) 

        if self.fno_block_precision == "mixed":
            x = x.astype(jnp.float16)

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = jnp.float16
        else:
            out_dtype = jnp.complex64

        out_fft = jnp.zeros(
            (batchsize, *fft_size, self.out_channels),
            dtype=out_dtype,
        )

        # if current modes are less than max, start indexing modes closer to the center of the weight tensor
        starts = [
            (max_modes - min(size, n_mode))
            for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)
        ]
        # if contraction is separable, weights have shape (channels, modes_x, ...)
        # otherwise they have shape (in_channels, out_channels, modes_x, ...)
        if self.separable:
            slices_w = [slice(None)]  # channels
        else:
            slices_w = [slice(None), slice(None)]  # in_channels, out_channels
        if self.complex_data:
            raise NotImplementedError(
                "complex_data=True is not supported in the JAX/NNX implementation yet."
            )
        else:
            # The last mode already has redundant half removed in real FFT
            slices_w += [
                slice(start // 2, -start // 2) if start else slice(start, None)
                for start in starts[:-1]
            ]
            slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)]

        # now convert slices_w to NHWC
        slices_w = slices_w[:1] + slices_w[2:] + slices_w[1:2]

        slices_w = tuple(slices_w)
        weight = self.weight[slices_w]     

        ### Pick the first n_modes modes of FFT signal along each dim
   
        slices_x = [slice(None)]  # Batch_size
        for all_modes, kept_modes in zip(fft_size, list(weight.shape[1:-1])):
            # After fft-shift, the 0th frequency is located at n // 2 in each direction
            # We select n_modes modes around the 0th frequency (kept at index n//2) by grabbing indices
            # n//2 - n_modes//2  to  n//2 + n_modes//2       if n_modes is even
            # n//2 - n_modes//2  to  n//2 + n_modes//2 + 1   if n_modes is odd
            center = all_modes // 2
            negative_freqs = kept_modes // 2
            positive_freqs = kept_modes // 2 + kept_modes % 2

            # this slice represents the desired indices along each dim
            slices_x += [slice(center - negative_freqs, center + positive_freqs)]

        if weight.shape[-2] < fft_size[-1]:
            slices_x[-1] = slice(None, weight.shape[-2])
        else:
            slices_x[-1] = slice(None)   

        slices_x.append(slice(None)) # channels
        slices_x = tuple(slices_x)

        out_fft = out_fft.at[slices_x].set(
            contract_dense(x[slices_x], weight, separable=self.separable)
        )      

        if self.resolution_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.resolution_scaling_factor)])

        if output_shape is not None:
            mode_sizes = output_shape  

        if self.order > 1:
            out_fft = jnp.fft.ifftshift(out_fft, axes=fft_dims[:-1])     

        if self.complex_data:
            raise NotImplementedError(
                "complex_data=True is not supported in the JAX/NNX implementation yet."
            )
        else:
            x = jnp.fft.irfftn(
                out_fft,
                s=mode_sizes,
                axes=fft_dims,
                norm=self.fft_norm
            )          

        if self.bias is not None:
            x = x + self.bias

        return x            
