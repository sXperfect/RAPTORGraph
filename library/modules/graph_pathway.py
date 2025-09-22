"""
Graph Pathway Layer Module.

This module provides a PyTorch layer designed to model relationships between
features (e.g., genes) and higher-level concepts (e.g., pathways) using
masked, grouped convolutions and optional interaction blocks.
"""
import typing as t
import enum
from enum import StrEnum

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import numpy as np
import torch
from torch import nn
import enum

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from .act_layer import get_act_layer, create_act_layer
from .norm import create_norm_layer, NormType

# =============================================================================
# CONFIGURATION ENUMS & CONSTANTS
# =============================================================================
class InteractionWeightSharing(enum.StrEnum):
    """Enumeration for interaction block weight sharing strategies."""
    SHARED = "shared"
    PER_PATHWAY = "per_pathway"

    @classmethod
    def _missing_(cls, value):
        """
        Handles conversion of uppercase or non-standard string values.
        
        This allows the enum to be created from strings like "SHARED" by
        converting them to lowercase. It raises an error for invalid values.
        """
        if isinstance(value, str):
            value_lower = value.lower()
            for member in cls:
                if member.value == value_lower:
                    return member
        valid_options = ", ".join(m.value for m in cls)
        raise ValueError(f"'{value}' is not a valid {cls.__name__}. Please use one of: {valid_options}")

class NormScope(StrEnum):
    """Enumeration for normalization scope."""
    PER_PATHWAY = "per_pathway"
    PER_GRAPH = "per_graph"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            value_lower = value.lower().replace('_', '')
            for member in cls:
                if member.value.replace('_', '') == value_lower:
                    return member
        valid_options = ", ".join(m.value for m in cls)
        raise ValueError(f"'{value}' is not a valid {cls.__name__}. Please use one of: {valid_options}")

#? Define the set of valid 1D normalization layers for this architecture.
VALID_1D_NORM_TYPES = {
    NormType.BATCH_NORM_1D,
    NormType.GROUP_NORM,
    NormType.GROUP_NORM_1,
    NormType.LAYER_NORM,
    NormType.RMS_NORM,
    NormType.SIMPLE_NORM,
}

#? --- Constants ---
DEFAULT_CLIP_VALUE = (0, 1)
DEFAULT_NUM_GRAPHS = 2
DEFAULT_ACTIVATION = 'silu'

# =============================================================================
# MODULE CLASSES
# =============================================================================
class GraphPathwayLayer(nn.Module):
    def __init__(self,
        gp_mask,
        num_graphs=DEFAULT_NUM_GRAPHS,
        graph_act_layer=DEFAULT_ACTIVATION,  # Added activation parameter
        graph_bias=True,
        graph_bn=False,
        pathway_bias=True,
        debug=False,
        clip_output=False,
        clip_value=DEFAULT_CLIP_VALUE,
    ):
        super().__init__()
        # self.register_buffer('mask', torch.from_numpy(gp_mask))

        #? Model dimensions
        num_genes = gp_mask.shape[1]  #? Number of expression features
        num_pathways = gp_mask.shape[0]  #? Number of pathways

        self.E = gp_mask.shape[1]  #? Number of expression features
        self.P = gp_mask.shape[0]  #? Number of pathways
        self.G = num_graphs  #? Number of groups per pathway

        #? Initialize linear layer
        graph_layers = []
        if debug:
            graph_fc_layer = nn.Linear(num_genes, num_graphs*num_pathways, bias=False)

        else:
            graph_fc_layer = nn.Linear(num_genes, num_graphs*num_pathways, bias=graph_bias)

        graph_layers.append(graph_fc_layer)

        if graph_act_layer is not None and not debug:
            graph_layers.append(
                get_act_layer(graph_act_layer)()
            )

        if graph_bn and not debug:
            graph_layers.append(nn.BatchNorm1d(self.P * self.G))

        self.graph_block = nn.Sequential(*graph_layers)

        #? Ensure mask has the correct shape
        expected_shape = (self.P, self.E)
        assert gp_mask.shape == (self.P, self.E), \
            f"Mask shape {gp_mask.shape} does not match expected shape ({expected_shape})"

        #? Transformation to copy mask num_graphs times
        gp_mask = np.repeat(gp_mask, self.G, axis=0)

        #? Convert mask to torch tensor and assign to weights
        # with torch.no_grad():
        mask_data = (
            torch.from_numpy(
                gp_mask.astype(float)
            )
            .type(torch.float32)
        )
        if debug:
            graph_fc_layer.weight.data = mask_data
        else:
            graph_fc_layer.weight.data *= mask_data

        #? Initialize pathway layer
        if debug:
            self.pathway_layer = nn.Conv1d(
                self.P,
                self.P,
                self.G,
                groups=self.P,
                bias=False
            )

            pathway_l_data = (
                torch.from_numpy(
                    np.ones(self.pathway_layer.weight.shape)
                )
                .type(torch.float32)
            )
            self.pathway_layer.weight.data = pathway_l_data
        else:
            self.pathway_layer = nn.Conv1d(
                self.P,
                self.P,
                self.G,
                groups=self.P,
                bias=pathway_bias
            )

        if clip_output and not debug:
            self.clip_layer = nn.Hardtanh(*clip_value)
        else:
            self.clip_layer = None

    def forward(self, x):
        batch_size = x.shape[0] #? x shape: (batch_size, E)
        out = self.graph_block(x)  #? Shape: (batch_size, P*G)
        bpg_tensor = out.T.reshape(batch_size, self.P, self.G)
        out = self.pathway_layer(bpg_tensor)  #? Shape: (batch_size, P, ...)
        out = out.reshape(batch_size, self.P)

        if self.clip_layer:
            out = self.clip_layer(out)

        return out

class PerGraphNorm(nn.Module):
    """A wrapper to apply 1D normalization per graph."""
    def __init__(self, num_pathways: int, num_graphs: int, norm_layer: nn.Module):
        super().__init__()
        self.num_pathways = num_pathways
        self.num_graphs = num_graphs
        self.norm_layer = norm_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #? Reshape from (N, P*G) to (N*P, G) to isolate graph features.
        x_reshaped = x.view(-1, self.num_graphs)
        #? Apply normalization across the G dimension for each pathway independently.
        x_norm = self.norm_layer(x_reshaped)
        #? Reshape back to the original (N, P*G).
        return x_norm.view(x.shape)

class GraphPathwayLayerV2(nn.Module):
    """
    A layer that models relationships between features (genes) and higher-level
    concepts (pathways) using masked linear layers, interaction blocks, and
    grouped convolutions.
    """

    def __init__(
        self,
        #? --- Data & Mask Configuration ---
        gp_mask: np.ndarray | None = None,
        num_pathways: int | None = None,
        num_genes: int | None = None,
        #? --- Graph Block Configuration ---
        num_graphs: int = 2,
        graph_act_layer: str | None = "silu",
        graph_bias: bool = True,
        graph_norm_layer: str | NormType | None = None,
        graph_norm_scope: str | NormScope = NormScope.PER_PATHWAY,
        #? --- VAE Mode ---
        is_variational: bool = False,
        #? --- Pathway Aggregation Configuration ---
        pathway_bias: bool = True,
        clip_output: bool = False,
        clip_value: tuple[float, float] = (0, 1),
    ):
        """
        Initializes the GraphPathwayLayerV2.

        Parameters
        ----------
        gp_mask : np.ndarray | None, optional
            A binary mask of shape (num_pathways, num_genes) defining the
            gene-pathway relationships. If None, a dense mask is assumed.
        num_pathways : int | None, optional
            The number of pathways (output dimensions). If None, it is inferred
            from `gp_mask`.
        num_genes : int | None, optional
            The number of genes (input dimensions). If None, it is inferred
            from `gp_mask`.
        num_graphs : int, optional
            The number of graph representations to learn for each pathway.
        graph_act_layer : str | None, optional
            The name of the activation function to use in the graph block.
        graph_bias : bool, optional
            Whether to include a bias term in the graph linear layer.
        graph_norm_layer : str | NormType | None, optional
            The name of the normalization layer to use. Must be a 1D type.
        graph_norm_scope : str | NormScope, optional
            The scope of normalization ('per_pathway' or 'per_graph').
        pathway_bias : bool, optional
            Whether to include a bias term in the pathway aggregation layer.
        clip_output : bool, optional
            If True, clips the final output of the layer.
        clip_value : tuple[float, float], optional
            The min and max values for clipping if `clip_output` is True.
        """
        super().__init__()
        self.num_graphs = num_graphs
        self.is_variational = is_variational

        #? --- Validate normalization layer compatibility ---
        if graph_norm_layer:
            norm_type = NormType(graph_norm_layer)
            if norm_type not in VALID_1D_NORM_TYPES:
                raise ValueError(
                    f"Invalid graph_norm_layer: '{graph_norm_layer}'. "
                    f"This architecture requires a 1D normalization layer. "
                    f"Valid options are: {[n.value for n in VALID_1D_NORM_TYPES]}"
                )

        #? --- Mask processing and validation ---
        self.final_mask = self._prepare_mask(
            gp_mask, 
            num_pathways, 
            num_genes, 
        )
        self.num_pathways, self.num_genes = self.final_mask.shape

        #? --- Graph block (masked linear layer + optional activation/normalization) ---
        self.graph_block = self._build_graph_block(
            graph_act_layer, 
            graph_bias, 
            graph_norm_layer, 
            graph_norm_scope
        )
        self._apply_mask_to_graph_layer()

        #? --- Pathway block (grouped convolution) ---
        out_channels = self.num_pathways * 2 if self.is_variational else self.num_pathways
        self.pathway_layer = nn.Conv1d(
            self.num_pathways, out_channels, self.num_graphs,
            groups=self.num_pathways, bias=pathway_bias
        )

        #? --- Optional output clipping ---
        self.clip_layer = nn.Hardtanh(*clip_value) if clip_output else nn.Identity()

    def reapply_mask(self):
        """Re-applies the mask to the graph layer's weights."""
        self._apply_mask_to_graph_layer()

    def _prepare_mask(
        self,
        gp_mask: np.ndarray | None,
        num_pathways: int | None,
        num_genes: int | None
    ) -> np.ndarray:
        """
        Validates and prepares the final mask for the layer.

        Parameters
        ----------
        gp_mask : np.ndarray | None
            The initial gene-pathway relationship mask.
        num_pathways : int | None
            The total number of pathways for the final mask.
        num_genes : int | None
            The total number of genes for the final mask.

        Returns
        -------
        np.ndarray
            The final, validated mask.
        """
        if gp_mask is not None:
            mask_rows, mask_cols = gp_mask.shape
            p_dim = num_pathways if num_pathways is not None else mask_rows
            e_dim = num_genes if num_genes is not None else mask_cols
            if mask_rows > p_dim or mask_cols > e_dim:
                raise ValueError(f"Provided gp_mask shape {gp_mask.shape} is larger than target dimensions ({p_dim}, {e_dim}).")
            final_mask = np.ones((p_dim, e_dim), dtype=np.float32)
            final_mask[:mask_rows, :mask_cols] = gp_mask
        elif num_pathways is not None and num_genes is not None:
            final_mask = np.ones((num_pathways, num_genes), dtype=np.float32)
        else:
            raise ValueError("Must provide either 'gp_mask' or both 'num_pathways' and 'num_genes'.")
        
        return final_mask

    def _build_graph_block(
        self,
        act_layer_name: str | None,
        bias: bool,
        norm_layer_name: str | None,
        norm_scope: str | NormScope
    ) -> nn.Sequential:
        """
        Constructs the sequential graph block.
        """
        layers = []
        self.graph_fc_layer = nn.Linear(self.num_genes, self.num_graphs * self.num_pathways, bias=bias)
        layers.append(self.graph_fc_layer)

        #? Conditionally add the chosen normalization layer.
        if norm_layer_name:
            scope = NormScope(norm_scope)
            if scope == NormScope.PER_PATHWAY:
                norm_dim = self.num_pathways * self.num_graphs
                norm_layer_obj = create_norm_layer(norm_layer_name, norm_dim)
                layers.append(norm_layer_obj)
            elif scope == NormScope.PER_GRAPH:
                norm_layer_obj = create_norm_layer(norm_layer_name, self.num_graphs)
                layers.append(
                    PerGraphNorm(
                        self.num_pathways, 
                        self.num_graphs, 
                        norm_layer_obj
                    )
                )
        
        if act_layer_name is not None:
            layers.append(
                create_act_layer(act_layer_name, inplace=True)
            )

        return nn.Sequential(*layers)

    def _apply_mask_to_graph_layer(self):
        """Applies the processed mask to the graph layer's weights."""
        #? The mask is repeated for each of the `num_graphs` to match the
        #? linear layer's weight shape.
        repeated_mask = np.repeat(self.final_mask, self.num_graphs, axis=0)
        mask_tensor = torch.from_numpy(repeated_mask).float().to(self.graph_fc_layer.weight.device)
        with torch.no_grad():
            #? This multiplication enforces the pathway structure by zeroing out
            #? connections between genes and unrelated pathways.
            self.graph_fc_layer.weight.data *= mask_tensor

    def forward(
        self, 
        x: torch.Tensor
    ) -> t.Union[torch.Tensor, t.Tuple[torch.Tensor, torch.Tensor]]:
        """Performs a forward pass through the layer."""
        batch_size = x.shape[0]
        
        graph_out = self.graph_block(x)
        conv_input = graph_out.view(batch_size, self.num_pathways, self.num_graphs)
        pathway_out = self.pathway_layer(conv_input)
        out = pathway_out.squeeze(-1)
        out = self.clip_layer(out)
        
        if self.is_variational:
            mu, var = torch.chunk(out, 2, dim=-1)
            return mu, var
        else:
            return out

class SharedInteractionBlock(nn.Module):
    """An interaction block with weights shared across all pathways."""
    def __init__(
        self,
        #? --- Architecture Configuration ---
        num_graphs: int,
        act_layer: str | None,
        use_norm: bool,
        use_residual: bool,
    ):
        super().__init__()
        self.use_residual = use_residual
        layers = [nn.Linear(num_graphs, num_graphs)]
        if use_norm:
            layers.append(nn.LayerNorm(num_graphs))
        if act_layer:
            layers.append(create_act_layer(act_layer, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.block(x)
        if self.use_residual:
            out = out + residual
        return out

class PathwaySpecificInteractionBlock(nn.Module):
    """An interaction block with a unique set of weights for each pathway."""
    def __init__(
        self,
        #? --- Architecture Configuration ---
        num_pathways: int,
        num_graphs: int,
        act_layer: str | None,
        use_norm: bool,
        use_residual: bool,
        bias: bool = True,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.weight = nn.Parameter(torch.Tensor(num_pathways, num_graphs, num_graphs))
        self.bias = nn.Parameter(torch.Tensor(num_pathways, num_graphs)) if bias else None

        self.norm = nn.LayerNorm(num_graphs) if use_norm else nn.Identity()
        self.act = get_act_layer(act_layer)() if act_layer else nn.Identity()
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.einsum('npg,pgk->npk', x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        out = self.norm(out)
        out = self.act(out)
        if self.use_residual:
            out = out + residual
        return out

class InteractingGraphPathwayLayer(GraphPathwayLayerV2):
    """
    An enhanced GraphPathwayLayer that includes intermediate blocks for
    interaction between graph representations within each pathway before
    the final aggregation.
    """

    def __init__(
        self,
        #? --- Data Configuration ---
        gp_mask: np.ndarray | None = None,
        num_pathways: int | None = None,
        num_genes: int | None = None,
        #? --- Base Architecture Configuration ---
        num_graphs: int = 2,
        graph_act_layer: str | None = "silu",
        graph_bias: bool = True,
        graph_norm_layer: str | NormType | None = None,
        graph_norm_scope: str | NormScope = NormScope.PER_PATHWAY,
        #? --- VAE Mode ---
        is_variational: bool = False,
        #? --- Interaction Block Configuration ---
        num_interaction_blocks: int = 1,
        interaction_weight_sharing: str | InteractionWeightSharing = 'per_pathway',
        interaction_act_layer: str | None = "silu",
        interaction_use_norm: bool = True,
        interaction_use_residual: bool = False,
        #? --- Pathway Aggregation Configuration ---
        pathway_bias: bool = True,
        clip_output: bool = False,
        clip_value: tuple[float, float] = (0, 1),
    ):
        super().__init__(
            #? --- Data & Mask Configuration ---
            gp_mask=gp_mask,
            num_pathways=num_pathways,
            num_genes=num_genes,
            #? --- Graph Block Configuration ---
            num_graphs=num_graphs,
            graph_act_layer=graph_act_layer,
            graph_bias=graph_bias,
            graph_norm_layer=graph_norm_layer,
            graph_norm_scope=graph_norm_scope,
            #? --- VAE Mode ---
            is_variational=is_variational,
            #? --- Pathway Aggregation Configuration ---
            pathway_bias=pathway_bias,
            clip_output=clip_output,
            clip_value=clip_value,
        )

        sharing_mode = InteractionWeightSharing(interaction_weight_sharing)

        interaction_blocks = []
        for _ in range(num_interaction_blocks):
            if sharing_mode == InteractionWeightSharing.SHARED:
                block = SharedInteractionBlock(
                    self.num_graphs,
                    interaction_act_layer,
                    interaction_use_norm,
                    interaction_use_residual,
                )
            else: #? PER_PATHWAY
                block = PathwaySpecificInteractionBlock(
                    self.num_pathways,
                    self.num_graphs,
                    interaction_act_layer,
                    interaction_use_norm,
                    interaction_use_residual,
                )
            interaction_blocks.append(block)

        self.interaction_block = nn.Sequential(*interaction_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        graph_out = self.graph_block(x)
        interaction_input = graph_out.view(
            batch_size,
            self.num_pathways,
            self.num_graphs
        )

        interaction_out = self.interaction_block(interaction_input)

        pathway_out = self.pathway_layer(interaction_out)
        out = pathway_out.squeeze(-1)

        out = self.clip_layer(out)
        if self.is_variational:
            mu, var = torch.chunk(out, 2, dim=-1)
            return mu, var
        else:
            return out
