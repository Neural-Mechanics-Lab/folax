"""
 Authors: Reza Najian Asl, https://github.com/RezaNajian
 Date: Dec, 2025
 License: FOL/LICENSE
"""

from typing import List, Optional, Union

Number = Union[float, int]

# JAX/Flax NNX implementation of fno_utils.
#
# Ported from the original PyTorch implementation:
#   Repository: https://github.com/neuraloperator/neuraloperator
#   File: neuralop/utils.py
#   Commit: 14c0f7320dc7c94e907a16fd276248df2d71407c (2025-11-14)
#   URL:
#     https://github.com/neuraloperator/neuraloperator/blob/14c0f7320dc7c94e907a16fd276248df2d71407c/neuralop/utils.py
#
# Original code copyright (c) 2023 NeuralOperator developers
# Licensed under the MIT License.
#
# Note:
#   The PyTorch implementation operates in NCHW (channels-first) format,
#   while JAX/Flax NNX uses NHWC (channels-last). This port includes
#   careful transformations between channel orders to preserve behavior.

def validate_scaling_factor(
    scaling_factor: Union[None, Number, List[Number], List[List[Number]]],
    n_dim: int,
    n_layers: Optional[int] = None,
) -> Union[None, List[float], List[List[float]]]:
    """
    Parameters
    ----------
    scaling_factor : None OR float OR list[float] Or list[list[float]]
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
        and all([isinstance(s, (float, int)) for s in scaling_factor])
    ):
        if n_layers is None and len(scaling_factor) == n_dim:
            # this is a dim-wise scaling
            return [float(s) for s in scaling_factor]
        return [[float(s)] * n_dim for s in scaling_factor]

    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (list)) for s in scaling_factor])
    ):
        s_sub_pass = True
        for s in scaling_factor:
            if all([isinstance(s_sub, (int, float)) for s_sub in s]):
                pass
            else:
                s_sub_pass = False
            if s_sub_pass:
                return scaling_factor

    return None

