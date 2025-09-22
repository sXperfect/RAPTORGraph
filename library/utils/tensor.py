"""
Utility functions for tensor operations.

This module contains functions for normalizing tensors and extracting elements
from tensors based on given indices.
"""

# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
from torch import Tensor
import torch.nn.functional as F

def l2norm(
    t: Tensor
) -> Tensor:
    """
    Normalizes the input tensor to unit length along the last dimension.

    Parameters
    ----------
    t : Tensor
        The input tensor to normalize.

    Returns
    -------
    Tensor
        The normalized tensor with unit length along the last dimension.

    Notes
    -----
    This function uses PyTorch's F.normalize along the last dimension, which
    """
    
    return F.normalize(t, dim=-1)

def extract(
    a: Tensor,
    t: Tensor,
    x_shape: tuple[int, ...]
) -> Tensor:
    """
    Extracts elements from a tensor based on indices in t.

    Parameters
    ----------
    a : Tensor
        Source tensor from which to extract elements.
    t : Tensor
        Indices tensor specifying positions to extract.
    x_shape : tuple[int, ...]
        Shape of the input tensor for dimensional alignment.

    Returns
    -------
    Tensor
        Extracted tensor with dimensions matching x_shape.

    Notes
    -----
    #? Uses gather to index into the tensor and reshapes the result to maintain
    #? dimensional consistency with the input tensor shape.
    
    This function gathers elements from the last dimension of tensor `a` using
    indices in `t`, then reshapes the output to have the same number of 
    dimensions as specified by `x_shape`.

    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))