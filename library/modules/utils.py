"""
Weight Initialization Utilities.

This module provides a set of standardized weight initialization functions
for PyTorch models, including strategies like Kaiming uniform and truncated
normal.
"""
from ..structs.base_enum import BaseStrEnum, OptionalBaseStrEnum

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch.nn as nn

# =============================================================================
# CONFIGURATION ENUMS
# =============================================================================
class WeightInitMode(BaseStrEnum):
    """Defines the weight initialization strategy."""
    KAIMING = "kaiming"
    TRUNC_NORMAL = "trunc_normal"

# =============================================================================
# INITIALIZATION HELPERS
# =============================================================================
def weights_init(
    m: nn.Module,
    mode: WeightInitMode | str | None = "kaiming",
    verbose: bool = False
) -> None:
    """
    Initializes weights of the network's layers using a specified strategy.

    This function is applied recursively to all modules in the network. It handles
    standard `nn.Linear`, `nn.Conv1d`, `nn.Embedding`, and various normalization
    layers explicitly. It also handles custom modules that follow the convention
    of having a `.weight` and `.bias` parameter.

    Parameters
    ----------
    m : nn.Module
        The module to initialize.
    mode : WeightInitMode | str, optional
        The initialization strategy to use ('kaiming' or 'trunc_normal').
        Defaults to WeightInitMode.KAIMING.
    verbose : bool, optional
        If True, prints the initialization of each layer to the console. Defaults to False.
    """
    #? --- Ensure mode is an enum for internal consistency ---
    mode = WeightInitMode(mode) if mode is not None else None

    if mode is not None:
        # --- Handle standard layer types explicitly ---
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            # Check for a special flag to force zero initialization
            if hasattr(m, '_zero_init') and m._zero_init:
                if verbose:
                    print(f"Zero-initializing {m.__class__.__name__} due to _zero_init flag.")
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                if verbose:
                    print(f"Initializing {m.__class__.__name__} with {mode.value} method.")
                if mode == WeightInitMode.TRUNC_NORMAL:
                    nn.init.trunc_normal_(m.weight, std=.02)
                else: # Default to Kaiming for linear/conv layers
                    if m.weight.dim() > 1:
                        nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            if verbose:
                print(f"Initializing {m.__class__.__name__} with ones (weight) and zeros (bias).")
            #? Norm layers are typically initialized to 1s for weight and 0s for bias
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Embedding):
            if verbose:
                print(f"Initializing {m.__class__.__name__} with standard normal distribution.")
            #? Embeddings are typically initialized with a standard normal distribution
            nn.init.normal_(m.weight, mean=0, std=1)


        #? This ensures modules like PathwaySpecificInteractionBlock are still initialized.
        elif hasattr(m, 'weight') and isinstance(m.weight, nn.Parameter):
            #? Check if it's a type we haven't handled yet
            if m.__class__.__name__ == 'DAGMALinear':
                #? Special case to initialize 
                nn.init.uniform_(m.weight, a=-0.1, b=0.1)
            elif not isinstance(m, (nn.Linear, nn.Conv1d, nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm, nn.Embedding)):
                if verbose:
                    print(f"Applying fallback init to {m.__class__.__name__} with {mode.value} method.")
                if mode == WeightInitMode.TRUNC_NORMAL:
                    nn.init.trunc_normal_(m.weight, std=.02)
                else: #? Default to Kaiming if appropriate
                    if m.weight.dim() > 1:
                        nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')

                if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, nn.Parameter):
                    nn.init.constant_(m.bias, 0)
