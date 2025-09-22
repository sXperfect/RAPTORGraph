"""
Normalization Layer Factory.

This module provides a centralized factory function for creating various PyTorch
normalization layers by name, leveraging the robust implementations from the
timm library.
"""

from enum import StrEnum

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch.nn as nn
from timm.layers.create_norm import (
    _NORM_MAP,
    _NORM_TYPES,
    get_norm_layer,
    create_norm_layer,
)
from timm.layers.norm import (
    GroupNorm,
    GroupNorm1,
    LayerNorm,
    LayerNorm2d,
    RmsNorm,
    RmsNorm2d,
    SimpleNorm,
    SimpleNorm2d,
)
from torchvision.ops.misc import FrozenBatchNorm2d

# =============================================================================
# CONFIGURATION ENUMS
# =============================================================================
class NormType(StrEnum):
    """
    Enumeration for supported normalization layer types.
    
    This enum is case-insensitive and supports common shorthands.
    """
    BATCH_NORM_1D = "batch_norm_1d"
    BATCH_NORM_2D = "batch_norm_2d"
    FROZEN_BATCH_NORM_2D = "frozen_batch_norm_2d"
    GROUP_NORM = "group_norm"
    GROUP_NORM_1 = "group_norm_1"
    LAYER_NORM = "layer_norm"
    LAYER_NORM_2D = "layer_norm_2d"
    RMS_NORM = "rms_norm"
    RMS_NORM_2D = "rms_norm_2d"
    SIMPLE_NORM = "simple_norm"
    SIMPLE_NORM_2D = "simple_norm_2d"

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return super()._missing_(value)

        #? Normalize the input string by making it lowercase and removing underscores.
        normalized_value = value.lower().replace('_', '')
        
        #? Handle common shorthands
        if normalized_value == "batchnorm":
            return cls.BATCH_NORM_2D # Default to 2D for simplicity
        
        #? Match against the normalized enum values.
        for member in cls:
            if member.value.replace('_', '') == normalized_value:
                return member
        
        valid_options = ", ".join(m.value for m in cls)
        raise ValueError(f"'{value}' is not a valid {cls.__name__}. Please use one of: {valid_options}")

#? A mapping from the enum to the actual layer class.
_NORM_LAYER_MAP = {
    NormType.BATCH_NORM_1D: nn.BatchNorm1d,
    NormType.BATCH_NORM_2D: nn.BatchNorm2d,
    NormType.FROZEN_BATCH_NORM_2D: FrozenBatchNorm2d,
    NormType.GROUP_NORM: GroupNorm,
    NormType.GROUP_NORM_1: GroupNorm1,
    NormType.LAYER_NORM: LayerNorm,
    NormType.LAYER_NORM_2D: LayerNorm2d,
    NormType.RMS_NORM: RmsNorm,
    NormType.RMS_NORM_2D: RmsNorm2d,
    NormType.SIMPLE_NORM: SimpleNorm,
    NormType.SIMPLE_NORM_2D: SimpleNorm2d,
}

# =============================================================================
# FACTORY FUNCTION
# =============================================================================
def create_norm_layer(
    layer_name: str | NormType,
    num_features: int,
    **kwargs
) -> nn.Module:
    """
    Creates an instance of a specified normalization layer.

    Parameters
    ----------
    layer_name : str | NormType
        The name of the normalization layer to create (e.g., "layernorm", "batch_norm_2d").
    num_features : int
        The number of features (channels for 2D, embedding dim for 1D) the
        normalization layer will operate on.
    **kwargs
        Additional keyword arguments to pass to the layer's constructor.

    Returns
    -------
    nn.Module
        An instance of the requested normalization layer.
    """
    try:
        norm_mode = NormType(layer_name)
        layer_class = _NORM_LAYER_MAP[norm_mode]
        
        #? LayerNorm and RMSNorm use `normalized_shape` as the arg name.
        #? BatchNorm and GroupNorm use `num_features` or `num_channels`.
        #? For simplicity, we pass it as the first positional argument.
        return layer_class(num_features, **kwargs)
    except (ValueError, KeyError) as e:
        raise ValueError(f"Failed to create normalization layer '{layer_name}'. Reason: {e}")

