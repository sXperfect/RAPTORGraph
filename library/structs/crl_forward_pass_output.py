"""Shared Data Structures for Causal Representation Learning (CRL) Models.

This module defines common data structures, such as the output of a model's
forward pass, that are used across the CRL components (e.g., models, losses,
training engines).
"""
# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import dataclasses
from dataclasses import dataclass

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch

# =============================================================================
# SHARED DATA STRUCTURES
# =============================================================================
@dataclass(frozen=False) #? frozen=False is default, but explicit here for clarity
class CRLForwardPassOutput:
    """
    Structured output from a CRL model's forward pass.
    Provides properties `kl_mu` and `kl_log_var` to intelligently select
    the correct distribution for KL divergence calculation.

    This dataclass has a fixed structure; new attributes cannot be added after
    instantiation. However, the values of existing attributes are mutable and
    can be updated via standard attribute access (e.g., `output.mu = ...`) or
    dictionary-style subscripting (e.g., `output["mu"] = ...`).

    Attributes
    ----------
    y_hat : torch.Tensor
        The final reconstructed output, typically after an intervention.
    x_recon : torch.Tensor
        The standard reconstruction of the input `x` (without intervention).
    z : torch.Tensor
        The latent sample used for decoding, including any interventions.
    mu : torch.Tensor | None
        The mean of the latent distribution (in VAE mode). Defaults to None.
    var : torch.Tensor | None
        The variance of the latent distribution (in VAE mode). Defaults to None.
    G : torch.Tensor | None
        The learned causal graph matrix (adjacency matrix), if applicable.
    """
    y_hat: torch.Tensor
    x_recon: torch.Tensor
    z: torch.Tensor
    z_obs: torch.Tensor | None = None
    mu: torch.Tensor | None = None
    log_var: torch.Tensor | None = None
    mu_obs: torch.Tensor | None = None
    log_var_obs: torch.Tensor | None = None
    G: torch.Tensor | None = None

    @property
    def kl_mu(self) -> torch.Tensor | None:
        """
        Returns the appropriate mu for KL divergence.
        Prioritizes the observational mu, falls back to the standard mu.
        """
        return self.mu_obs if self.mu_obs is not None else self.mu

    @property
    def kl_log_var(self) -> torch.Tensor | None:
        """
        Returns the appropriate log_var for KL divergence.
        Prioritizes the observational log_var, falls back to the standard log_var.
        """
        return self.log_var_obs if self.log_var_obs is not None else self.log_var

    #? --- Dictionary-like behavior for convenience & backward compatibility ---
    def __getitem__(self, key: str) -> torch.Tensor | None:
        """Allows dictionary-style reading of fields (e.g., `output["mu"]`)."""
        try:
            return getattr(self, key)
        except AttributeError:
            #? Raise a KeyError to perfectly mimic dictionary behavior.
            raise KeyError(f"'{key}' is not a valid field in {self.__class__.__name__}")

    def __setitem__(self, key: str, value: torch.Tensor | None) -> None:
        """
        Allows dictionary-style setting of existing fields.

        This method enforces structural immutability: it prevents adding new
        keys to the object, raising a KeyError if an unknown key is used.
        However, it allows modifying the values of existing fields.
        """
        if not hasattr(self, key):
            #? This check is the guard that prevents adding new items.
            raise KeyError(
                f"Cannot add new key '{key}'. "
                f"{self.__class__.__name__} has a fixed set of fields."
            )
        #? If the key exists, setting the attribute is allowed.
        setattr(self, key, value)

    def __iter__(self):
        """Allows iterating over the defined field names."""
        for field in dataclasses.fields(self):
            yield field.name

    def __len__(self) -> int:
        """Returns the number of defined fields."""
        return len(dataclasses.fields(self))

    def __str__(self) -> str:
        """Provides a user-friendly summary of the output tensors."""
        def format_tensor(name: str, tensor: torch.Tensor | None) -> str:
            if tensor is None:
                return f"  - {name}: None"
            return (
                f"  - {name}: shape={list(tensor.shape)}, "
                f"device={tensor.device}, dtype={tensor.dtype}"
            )

        parts = [
            f"{self.__class__.__name__}:",
            *(format_tensor(name, getattr(self, name)) for name in self)
        ]
        return "\n".join(parts)