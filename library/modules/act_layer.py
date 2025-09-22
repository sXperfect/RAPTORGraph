"""
Activation Layer Factory.

This module provides a centralized factory for creating PyTorch activation
layers by name, including a custom Magnitude-Preserving SiLU (MPSiLU).
It is adapted from the timm library's practices for creating layers dynamically.
"""
import typing as t

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers.create_act import _ACT_LAYER_DEFAULT

# =============================================================================
# CUSTOM ACTIVATION LAYERS
# =============================================================================

class MPSiLU(nn.Module):
    """
    A magnitude-preserving Sigmoid Linear Unit (SiLU) activation layer.

    This layer implements the SiLU function, `x * sigmoid(x)`, and then scales
    the output by a constant factor (1/0.596) to preserve its magnitude.
    """
    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the MPSiLU layer.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after applying the scaled SiLU function.
        """
        return F.silu(x) / 0.596

#? --- Register the custom activation layer ---
_ACT_LAYER_DEFAULT['mpsilu'] = MPSiLU

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def get_act_layer(
    name: str | t.Type[nn.Module] = 'relu'
) -> t.Type[nn.Module] | None:
    """
    Fetches an activation layer class by its string name.

    Parameters
    ----------
    name : str | t.Type[nn.Module], optional
        The name of the activation layer or the class itself. Defaults to 'relu'.

    Returns
    -------
    t.Type[nn.Module] | None
        The activation layer class, or None if the name is empty.
    """
    if not name:
        return None
    if not isinstance(name, str):
        #? If it's already a class, just return it
        return name
    
    name = name.lower()
    act_layer = _ACT_LAYER_DEFAULT.get(name)
    if act_layer is None:
        raise ValueError(f"Unknown activation type ({name})")
    return act_layer

def create_act_layer(
    name: str | t.Type[nn.Module],
    inplace: bool | None = None,
    **kwargs
) -> nn.Module | None:
    """
    Creates an instance of an activation layer.

    Parameters
    ----------
    name : str | t.Type[nn.Module]
        The name of the activation layer or the class itself.
    inplace : bool | None, optional
        Whether the operation should be performed in-place, if supported.
        Defaults to None.
    **kwargs
        Additional keyword arguments for the layer's constructor.

    Returns
    -------
    nn.Module | None
        An instantiated activation layer, or None if the name is empty.
    """
    act_layer = get_act_layer(name)
    if act_layer is None:
        return None

    #? --- Attempt to instantiate with inplace argument ---
    if inplace is not None:
        try:
            return act_layer(inplace=inplace, **kwargs)
        except TypeError:
            #? Fallback for layers that don't support `inplace`
            pass
    
    #? --- Instantiate without inplace argument ---
    return act_layer(**kwargs)
