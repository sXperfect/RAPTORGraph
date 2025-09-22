"""
PyTorch implementation of a conditional latent modulator network.
This module provides a unified, highly flexible class for modulating both
deterministic and probabilistic latent variables with built-in output scaling
for improved training stability.
"""
# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import enum
import typing as t

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from ..structs import BaseStrEnum, OptionalBaseStrEnum
from ..consts import AttentionBackend
from .norm import create_norm_layer, NormType
from .act_layer import create_act_layer
from .residual_mlp import BasicResidualMLPBlock

# =============================================================================
# CONSTANTS
# =============================================================================

class ModulatorMode(BaseStrEnum):
    """Enumeration for the modulator's operational mode."""
    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"

class StateContextMode(OptionalBaseStrEnum):
    """Enumeration for state context handling modes."""
    MLP = "mlp"
    RAW = "raw"
    NONE = "none"

class FusionMode(BaseStrEnum):
    """Enumeration for feature fusion strategies."""
    CONCAT = "concat"

# =============================================================================
# IMPLEMENTATION
# =============================================================================

class LatentModulator(nn.Module):
    """
    A unified, flexible network for modulating latent variable.

    This module provides a class that can modulate both deterministic and probabilistic
    latent variables based on condition vectors. It includes a scaled tanh activation
    on its output to ensure the predicted modulations (deltas) are bounded, which
    significantly improves training stability.
    """
    
    def __init__(
        self,
        #? --- Main Dimension Configuration ---
        output_dim: int,
        #? --- Condition Encoder (Perturbation) Config ---
        bool_input_dim: int,
        bool_embedding_dim: int | None = None,
        #? --- Main Operating Mode ---
        mode: str | ModulatorMode = 'deterministic',
        #? --- State Context Config ---
        state_context_mode: str | StateContextMode | None = 'mlp',
        gene_input_dim: int | None = None,
        gene_hidden_dim: int | None = None,
        gene_output_dim: int | None = None,
        gene_num_layers: int = 3,
        detach_context: bool = False,
        #? --- Fusion & Head Config ---
        fusion_mode: str | FusionMode = 'concat',
        tower_norm_layer: str | NormType | None = None,
        head_hidden_dim: int | None = None,
        head_num_layers: int = 2,
        zero_init_head: bool = True,
        #? --- Shared Config ---
        act_layer: str = 'silu',
        use_residual: bool = True,
    ):
        """
        A unified, flexible network for modulating latent variables.

        This module provides a class that can modulate both deterministic and probabilistic
        latent variables based on condition (perturbation) vectors and optionally state context.

        The parameters are organized into several logical groups for clarity:
        
        - **Main Dimension Configuration**: Core dimensionality parameters
        - **Condition Encoder (Perturbation) Config**: Perturbation vector processing
        - **Main Operating Mode**: Fundamental behavior configuration
        - **State Context Config**: Current latent state handling
        - **Fusion & Head Config**: Feature combination and output prediction
        - **Shared Config**: Common activation and architectural settings

        Parameters
        ----------
        output_dim : int
            The dimension of the output latent vector (either deterministic or the mean/log variance for probabilistic).
            Must be >= `bool_input_dim`.
        bool_input_dim : int
            The dimension of the boolean input vector (perturbation vector) which is one-hot or multi-hot encoded.
        bool_embedding_dim : int | None, optional
            The embedding dimension for the condition (perturbation) vector. If None, defaults to `bool_input_dim`.
        mode : str | ModulatorMode, default='deterministic'
            The operational mode of the modulator. Options:
            - 'deterministic': Directly modulates latent vector `z`
            - 'probabilistic': Modulates mean `mu` and log variance `log_var`
        state_context_mode : str | StateContextMode | None, optional
            How to handle the state context (current latent state). Options:
            - 'mlp': Uses MLP to encode control input `x_ctrl` (requires `gene_input_dim`)
            - 'raw': Uses raw latent state directly (`mu`/`log_var` or `z`)
            - 'none': No state context used
        gene_input_dim : int | None, optional
            Required if `state_context_mode='mlp'`. Input dimension for state context MLP.
        gene_hidden_dim : int | None, optional
            Hidden dimension for state context MLP. If None, defaults to `gene_input_dim`.
        gene_output_dim : int | None, optional
            Output dimension for the state context MLP.
        gene_num_layers : int, default 3
            Number of layers in the state context MLP.
        detach_context : bool, default False
            If True, detaches the context tensor before it is used as input
            to the modulator. This prevents gradients from the intervention
            loss from flowing back to the encoder.
        fusion_mode : str | FusionMode, default 'concat'
            Feature fusion strategy ('concat').
        fusion_num_heads : int, default 4
            Number of attention heads for cross-attention fusion.
        use_basic_attn : bool, default=True
            Whether to use basic cross-attention block. If False, uses advanced block with `attn_backend` support.
        attn_backend : str | AttentionBackend, default='pytorch'
            Attention backend for advanced cross-attention (only used when `use_basic_attn=False`).
        tower_norm_layer : str | NormType | None, optional
            Normalization layer type applied to features before fusion. If None, uses identity normalization.
        head_hidden_dim : int | None, optional
            Hidden dimension for prediction head MLP. If None, defaults to fused feature dimension.
        head_num_layers : int, default=2
            Number of layers in prediction head MLP.
        zero_init_head : bool, default=True
            Whether to zero-initialize the final linear layer of the head. Improves training stability.
        act_layer : str, default='silu'
            Activation function for MLP blocks (see `create_act_layer` for valid options).
        use_residual : bool, default=True
            Whether to use residual connections in MLP blocks.

        Notes
        -----
        **Forward Method Requirements**:
        - `perts_vec` must always be provided (shape: [batch_size, bool_input_dim])
        - In 'mlp' context mode: `x_ctrl` must be provided
        - In 'raw' probabilistic mode: `mu` and `log_var` must be provided
        - In 'raw' deterministic mode: `z` must be provided

        **Output Behavior**:
        - Probabilistic mode: Returns tuple `(mu_final, log_var_final)` (both shape: [batch_size, output_dim])
        - Deterministic mode: Returns single tensor `z_final` (shape: [batch_size, output_dim])

        **Fusion Constraints**:
        - Cross-attention fusion requires `bool_embedding_dim == feature_b_dim`
        - Masking ensures modulations only affect perturbed features (via `perts_vec`)

        **Implementation Details**:
        - Uses intervention masking to apply changes only to perturbed features
        - Head outputs delta values that are added to original latents
        - Zero-initialization of head final layer prevents initial disruption

        Examples
        --------

        See Also
        --------
        BasicResidualMLPBlock : State context encoder implementation
        BasicCrossAttentionBlock : Default attention fusion implementation
        CrossAttentionBlock : Advanced attention fusion implementation
        """
        super().__init__()
        self.mode = ModulatorMode(mode)
        self.state_context_mode = StateContextMode(state_context_mode)
        self.fusion_mode = FusionMode(fusion_mode)
        self.detach_context = detach_context

        if output_dim < bool_input_dim:
            raise ValueError(f"output_dim ({output_dim}) must be >= bool_input_dim ({bool_input_dim}).")
        self.output_dim = output_dim

        if bool_embedding_dim is None: bool_embedding_dim = bool_input_dim

        #? --- Determine dimensions based on context mode ---
        if self.state_context_mode == StateContextMode.MLP:
            if gene_input_dim is None: raise ValueError("gene_input_dim is required for 'mlp' context mode.")
            if gene_hidden_dim is None: gene_hidden_dim = gene_input_dim
            if gene_output_dim is None: gene_output_dim = gene_input_dim
            feature_b_dim = gene_output_dim
        elif self.state_context_mode == StateContextMode.RAW:
            feature_b_dim = output_dim * 2 if self.mode == ModulatorMode.PROBABILISTIC else output_dim
        else: #? NONE
            feature_b_dim = 0

        #? --- Handle Fusion Logic ---
        if feature_b_dim > 0:
            if self.fusion_mode == FusionMode.CONCAT:
                fused_dim = bool_embedding_dim + feature_b_dim
        else:
            fused_dim = bool_embedding_dim

        if head_hidden_dim is None:
            head_hidden_dim = fused_dim

        #? --- Build Modules ---
        self.condition_encoder = nn.Embedding(num_embeddings=bool_input_dim, embedding_dim=bool_embedding_dim)

        if self.state_context_mode == StateContextMode.MLP:
            self.state_context_encoder = BasicResidualMLPBlock(in_dim=gene_input_dim, hidden_dim=gene_hidden_dim, out_dim=gene_output_dim, num_layers=gene_num_layers, act_layer=act_layer, use_residual=use_residual)
        else:
            self.state_context_encoder = None


        if tower_norm_layer and feature_b_dim > 0:
            self.norm_a = create_norm_layer(tower_norm_layer, num_features=bool_embedding_dim)
            self.norm_b = create_norm_layer(tower_norm_layer, num_features=feature_b_dim)
        else:
            self.norm_a = nn.Identity()
            self.norm_b = nn.Identity()

        if self.mode == ModulatorMode.PROBABILISTIC:
            head_output_dim = self.output_dim * 2 #? delta_mu, delta_log_var
        else: #? ModulatorMode.DETERMINISTIC
            head_output_dim = self.output_dim

        head_final_layer = nn.Linear(head_hidden_dim, head_output_dim)
        if zero_init_head:
            head_final_layer._zero_init = True

        self.head_block = nn.Sequential(
            BasicResidualMLPBlock(
                in_dim=fused_dim,
                hidden_dim=head_hidden_dim,
                out_dim=head_hidden_dim,
                num_layers=head_num_layers,
                act_layer=act_layer,
                use_residual=use_residual
            ),
            head_final_layer
        )


    def forward(self,
        perts_vec: torch.Tensor,
        mu: torch.Tensor | None = None,
        log_var: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        x_ctrl: torch.Tensor | None = None,
        detach_context: bool | None = None,
    ) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass that modulates latent variables based on perturbation conditions.

        Applies learned transformations to latent variables using condition vectors and
        optional state context. The behavior depends on the configured modulator mode
        and state context handling.

        Parameters
        ----------
        perts_vec : torch.Tensor
            Perturbation vector of shape [batch_size, bool_input_dim], indicating which
            features are perturbed (typically one-hot or multi-hot encoded).
        mu : torch.Tensor | None, optional
            Mean of the latent distribution (required in probabilistic mode with 'raw' context).
        log_var : torch.Tensor | None, optional
            Log variance of the latent distribution (required in probabilistic mode with 'raw' context).
        z : torch.Tensor | None, optional
            Deterministic latent vector (required in deterministic mode with 'raw' context).
        x_ctrl : torch.Tensor | None, optional
            Control input for state context MLP (required when state_context_mode='mlp').
        detach_context : bool | None, optional
            If provided, this overrides the instance's `self.detach_context`
            setting for this forward pass.

        Returns
        -------
        torch.Tensor | tuple[torch.Tensor, torch.Tensor]
            - In deterministic mode: Modulated latent vector `z_final` [batch_size, output_dim]
            - In probabilistic mode: Tuple `(mu_final, log_var_final)` 
              (both [batch_size, output_dim])

        Raises
        ------
        ValueError
            If required inputs are missing based on current configuration:
            - `x_ctrl` not provided when state_context_mode='mlp'
            - `mu`/`log_var` not provided in probabilistic 'raw' mode
            - `z` not provided in deterministic 'raw' mode

        Notes
        -----
        **Masking Behavior**:
        - Only perturbed features (indicated by `perts_vec`) receive modulation
        - Non-perturbed features retain their original values
        - Mask is automatically padded to match output dimension

        **Context Handling**:
        - 'mlp' mode: Processes `x_ctrl` through state context encoder
        - 'raw' mode: Uses provided latent variables directly
        - 'none' mode: Ignores state context entirely

        **Modulation Process**:
        1. Encodes perturbation vector into embedding space
        2. Processes state context (if applicable)
        3. Fuses condition and context features
        4. Predicts delta values through head network
        5. Applies masked addition to original latents

        Examples
        --------
        """

        #? --- Determine if context should be detached for this forward pass ---
        should_detach = self.detach_context if detach_context is None else detach_context
        if should_detach:
            if mu is not None: mu = mu.detach()
            if log_var is not None: log_var = log_var.detach()
            if z is not None: z = z.detach()
            if x_ctrl is not None: x_ctrl = x_ctrl.detach()
        
        #? 1. Get features from condition encoder
        cond_embedding = torch.matmul(perts_vec, self.condition_encoder.weight)

        #? 2. Get features from state context
        if self.state_context_mode == StateContextMode.MLP:
            assert x_ctrl is not None, \
                "x_ctrl is required for 'mlp' context mode."
            context_embedding = self.state_context_encoder(x_ctrl)
        elif self.state_context_mode == StateContextMode.RAW:
            if self.mode == ModulatorMode.PROBABILISTIC:
                if mu is None or log_var is None:
                    raise ValueError("mu and log_var are required for 'raw' probabilistic mode.")
                context_embedding = torch.cat([mu, log_var], dim=-1)
            else: #? ModulatorMode.DETERMINISTIC
                if z is None:
                    raise ValueError("z is required for 'raw' deterministic mode.")
                context_embedding = z
        else: #? NONE
            context_embedding = None

        #? 3. Normalize and Fuse features
        cond_embedding = self.norm_a(cond_embedding)
        if context_embedding is not None:
            context_embedding = self.norm_b(context_embedding)
            if self.fusion_mode == FusionMode.CONCAT:
                fused_embedding = torch.cat([cond_embedding, context_embedding], dim=1)
        else: #? No context to fuse
            fused_embedding = cond_embedding

        #? 4. Predict modulation
        modulation = self.head_block(fused_embedding)

        #? 5. Apply modulation based on intervention mask
        mask = F.pad(perts_vec, (0, self.output_dim - perts_vec.shape[-1]), "constant", 0)

        if self.mode == ModulatorMode.PROBABILISTIC:
            if mu is None or log_var is None:
                raise ValueError("`mu` and `log_var` must be provided for probabilistic mode.")

            delta_mu, delta_log_var = modulation.chunk(2, dim=-1)

            mu_final = mu + (mask * delta_mu)
            log_var_final = log_var + (mask * delta_log_var)

            return mu_final, log_var_final

        elif self.mode == ModulatorMode.DETERMINISTIC:

            if z is None:
                raise ValueError("`z` must be provided for deterministic mode.")

            #? The head outputs a delta for z
            delta_z = modulation
            z_final = z + (mask * delta_z)

            return z_final

        else:
            raise ValueError(f"Unsupported modulator mode: {self.mode}")
