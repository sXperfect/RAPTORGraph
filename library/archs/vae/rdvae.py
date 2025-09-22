"""
GraphPathway Variational Autoencoder (VAE) Architecture with Causal Intervention.

This module defines a VAE that uses a GraphPathwayLayer as its encoder and
integrates a LatentModulator to perform causal interventions in the latent space.
"""
# =============================================================================
# STANDARD LIBRARY IMPORTS
# =============================================================================
import typing as t
from functools import partial

# =============================================================================
# THIRD-PARTY IMPORTS
# =============================================================================
import torch
import torch.nn as nn
import numpy as np

# =============================================================================
# LOCAL APPLICATION IMPORTS
# =============================================================================
from ...structs import (
    CRLForwardPassOutput,
    BaseStrEnum,
)
from ...modules.act_layer import create_act_layer
from ...modules.graph_pathway import (
    GraphPathwayLayerV2,
    InteractingGraphPathwayLayer,
    InteractionWeightSharing,
    NormScope,
)
from ...modules.norm import NormType, create_norm_layer
from ...datasets.pathway_handler import create_deconfounding_pathway_mask
from ...modules.utils import (
    weights_init,
    WeightInitMode,
)
from ...modules.latent_modulator import (
    LatentModulator,
    ModulatorMode,
    FusionMode,
    StateContextMode,
    AttentionBackend
)
from ...modules.dagma_linear import DAGMALinear

# =============================================================================
# CONFIGURATION ENUMS
# =============================================================================
class EncoderType(BaseStrEnum):
    """Defines the type of encoder architecture to use."""
    GRAPH_PATHWAY = "graph_pathway"
    INTERACTING_GRAPH_PATHWAY = "interacting_graph_pathway"

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
class RaptorGraphVAEArch(nn.Module):
    """
    A VAE with a GraphPathway encoder and an integrated LatentModulator for
    causal interventions.
    """
    def __init__(
        self,
        #? --- Core Dimensions & Masks ---
        gene_names: t.List[str],
        perturb_gene_names: t.List[str],
        num_pathways: int | None = None,
        gp_mask: np.ndarray | None = None,
        gene_mask: torch.Tensor | None = None,
        #? --- Main Architecture Configuration ---
        mode: str | ModulatorMode = 'probabilistic',
        encoder_type: str | EncoderType = 'graph_pathway',
        hids: int = 128,
        decoder_act_layer: str = "silu",
        weight_init_mode: str | WeightInitMode = 'trunc_normal',
        #? --- Graph Pathway Encoder Arguments ---
        num_graphs: int = 2,
        graph_act_layer: str | None = "silu",
        graph_bias: bool = True,
        graph_norm_layer: str | NormType | None = None,
        graph_norm_scope: str | NormScope = "per_pathway",
        pathway_bias: bool = True,
        clip_output: bool = False,
        clip_value: t.Tuple[float, float] = (0, 1),
        #? --- Interacting Encoder Specific Arguments ---
        interaction_weight_sharing: str | InteractionWeightSharing = "per_pathway",
        num_interaction_blocks: int = 1,
        interaction_act_layer: str | None = "silu",
        interaction_use_norm: bool = True,
        interaction_use_residual: bool = False,
        #? --- Latent Modulator Arguments ---
        modulator_bool_embedding_dim: int | None = None,
        modulator_state_context_mode: str | StateContextMode | None = 'mlp',
        modulator_gene_hidden_dim: int | None = None,
        modulator_gene_output_dim: int | None = None,
        modulator_gene_num_layers: int = 3,
        modulator_detach_context: bool = False,
        modulator_fusion_mode: str | FusionMode = 'concat',
        modulator_fusion_num_heads: int = 4,
        modulator_use_basic_attn: bool = True,
        modulator_attn_backend: str | AttentionBackend = 'pytorch',
        modulator_tower_norm_layer: str | NormType | None = None,
        modulator_head_hidden_dim: int | None = None,
        modulator_head_num_layers: int = 2,
        modulator_zero_init_head: bool = True,
        modulator_act_layer: str = 'silu',
        modulator_use_residual: bool = True,
        #? --- DAGMA LInear Arguments ---
        dagma_w_threshold: float | None = 0.3,
        dagma_force_dag: bool = False,
        dagma_norm_layer: str | NormType | None = None,
        #?
        out_nonneg: bool = False,
    ):
        super().__init__()

        #? --- Store all configuration parameters as instance attributes ---
        self.mode = ModulatorMode(mode)
        self.gene_names = gene_names
        self.perturb_gene_names = perturb_gene_names
        self.gp_mask = gp_mask
        self.gene_mask = gene_mask
        self.encoder_type = EncoderType(encoder_type)
        self.hids = hids
        self.decoder_act_layer = decoder_act_layer
        self.weight_init_mode = WeightInitMode(weight_init_mode)
        self.num_graphs = num_graphs
        self.graph_act_layer = graph_act_layer
        self.graph_bias = graph_bias
        self.graph_norm_layer = graph_norm_layer
        self.graph_norm_scope = graph_norm_scope
        self.pathway_bias = pathway_bias
        self.clip_output = clip_output
        self.clip_value = clip_value
        self.interaction_weight_sharing = interaction_weight_sharing
        self.num_interaction_blocks = num_interaction_blocks
        self.interaction_act_layer = interaction_act_layer
        self.interaction_use_norm = interaction_use_norm
        self.interaction_use_residual = interaction_use_residual
        self.modulator_state_context_mode = StateContextMode(modulator_state_context_mode)

        self.out_nonneg = out_nonneg

        #? --- Set and validate the number of pathways ---
        if num_pathways is None:
            self.num_pathways = self.min_num_pathways
        else:
            if num_pathways < self.min_num_pathways:
                raise ValueError(
                    f"num_pathways ({num_pathways}) cannot be less than the "
                    f"number of perturbation genes ({self.min_num_pathways})."
                )
            self.num_pathways = num_pathways

        if self.gp_mask is not None and self.gp_mask.shape[0] < self.num_pathways:
            raise ValueError(
                f"The number of rows in gp_mask ({self.gp_mask.shape[0]}) must be "
                f"greater than or equal to num_pathways ({self.num_pathways})."
            )

        #? --- Build network components ---
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        #? --- DAGMA Normalization Layer Setup ---
        VALID_DAGMA_NORM_TYPES = {NormType.LAYER_NORM, NormType.RMS_NORM}
        if dagma_norm_layer is not None:
            if NormType(dagma_norm_layer) not in VALID_DAGMA_NORM_TYPES:
                raise ValueError(
                    f"Invalid dagma_norm_layer: '{dagma_norm_layer}'. "
                    f"Valid options for DAGMA norm are: {[n.value for n in VALID_DAGMA_NORM_TYPES]}"
                )
            self.dagma_norm = create_norm_layer(dagma_norm_layer, self.num_pathways)
        else:
            self.dagma_norm = nn.Identity()
        #? --- Build the Latent Modulator for interventions ---
        self.modulator = LatentModulator(
            #? --- Main Dimension Configuration ---
            output_dim=self.num_pathways,
            #? --- Condition Encoder (Perturbation) Config ---
            bool_input_dim=len(self.perturb_gene_names),
            bool_embedding_dim=modulator_bool_embedding_dim,
            #? --- Main Operating Mode ---
            mode=self.mode,
            #? --- State Context Config ---
            state_context_mode=self.modulator_state_context_mode,
            gene_input_dim=self.num_genes,
            gene_hidden_dim=modulator_gene_hidden_dim,
            gene_output_dim=modulator_gene_output_dim,
            gene_num_layers=modulator_gene_num_layers,
            detach_context=modulator_detach_context,
            #? --- Fusion & Head Config ---
            fusion_mode=modulator_fusion_mode,
            tower_norm_layer=modulator_tower_norm_layer,
            head_hidden_dim=modulator_head_hidden_dim,
            head_num_layers=modulator_head_num_layers,
            zero_init_head=modulator_zero_init_head,
            #? --- Shared Config ---
            act_layer=modulator_act_layer,
            use_residual=modulator_use_residual,
        )

        self.dagma_layer = DAGMALinear(
            #? --- Model Configuration ---
            d=self.num_pathways,
            w_threshold=dagma_w_threshold,
            force_dag=dagma_force_dag,
        )

        self.clip_layer = nn.ReLU() if self.out_nonneg else nn.Identity()

        #? --- Initialize weights ---
        init_func = partial(
            weights_init,
            mode=self.weight_init_mode
        )
        self.apply(init_func)

        #? --- Apply masks AFTER weight initialization ---
        self.encoder.reapply_mask()
        if self.gene_mask is not None:
            self._apply_decoder_mask()

    @property
    def num_genes(self):
        return len(self.gene_names)

    @property
    def dim(self):
        return self.num_genes

    @property
    def min_num_pathways(self):
        return len(self.perturb_gene_names)

    def _build_encoder(self) -> nn.Module:
        """Factory method to construct the appropriate encoder."""
        if self.gp_mask is None:
            active_gp_mask = create_deconfounding_pathway_mask(
                self.gene_names,
                self.perturb_gene_names,
                self.num_pathways,
            )
        else:
            active_gp_mask = self.gp_mask

        if self.gene_mask is not None:
            gene_mask_np = self.gene_mask.numpy().reshape(1, -1)
            active_gp_mask = active_gp_mask * gene_mask_np

        common_args = {
            "num_genes": self.dim,
            "gp_mask": active_gp_mask,
            "num_pathways": self.num_pathways,
            "num_graphs": self.num_graphs,
            "graph_act_layer": self.graph_act_layer,
            "graph_bias": self.graph_bias,
            "graph_norm_layer": self.graph_norm_layer,
            "graph_norm_scope": self.graph_norm_scope,
            "pathway_bias": self.pathway_bias,
            "clip_output": self.clip_output,
            "clip_value": self.clip_value,
            "is_variational": self.is_variational,
        }

        if self.encoder_type == EncoderType.GRAPH_PATHWAY:
            return GraphPathwayLayerV2(**common_args)

        if self.encoder_type == EncoderType.INTERACTING_GRAPH_PATHWAY:
            return InteractingGraphPathwayLayer(
                **common_args,
                interaction_weight_sharing=self.interaction_weight_sharing,
                num_interaction_blocks=self.num_interaction_blocks,
                interaction_act_layer=self.interaction_act_layer,
                interaction_use_norm=self.interaction_use_norm,
                interaction_use_residual=self.interaction_use_residual,
            )

        raise ValueError(f"Unsupported encoder_type: {self.encoder_type}")

    def _build_decoder(self) -> nn.Module:
        """Constructs the decoder as a sequential MLP."""
        return nn.Sequential(
            nn.Linear(self.num_pathways, self.hids),
            create_act_layer(self.decoder_act_layer, inplace=True),
            nn.Linear(self.hids, self.dim),
            create_act_layer(self.decoder_act_layer, inplace=True),
        )

    def _apply_decoder_mask(self):
        """Applies the winsorizing mask to the final decoder layer."""
        with torch.no_grad():
            final_linear_layer = self.decoder[2]
            final_linear_layer.weight.data[~self.gene_mask, :] = 0
            if final_linear_layer.bias is not None:
                final_linear_layer.bias.data[~self.gene_mask] = 0

    @property
    def is_variational(self) -> bool:
        return True

    @property
    def is_causal(self) -> bool:
        return True

    @property
    def is_dagma(self) -> bool:
        return True

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Performs the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodes a latent vector back into the data space."""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        c1: torch.Tensor = None,
        c2: torch.Tensor = None,
        num_interv: int = None,
        temp: float = 1.0,
        use_causal_layer: bool = True,
        return_dict: bool = True,
    ) -> CRLForwardPassOutput | t.List[torch.Tensor]:
        """
        Performs a forward pass through the autoencoder.

        If a perturbation vector is provided, it applies a causal intervention
        in the latent space according to the specified `mode`.
        """
        #? --- 1. Encode to get the initial latent distribution ---
        mu_obs, log_var_obs = self.encoder(x)
        z_obs = self.reparameterize(mu_obs, log_var_obs)

        #? --- 2. Handle intervention based on the selected mode ---
        z_final, mu_final, log_var_final = None, None, None

        if self.modulator.state_context_mode != StateContextMode.RAW:
            context_x = x
        else:
            context_x = None

        if c1 is not None:
            if num_interv <= 1:
                c = c1
            else:
                c = c1 + c2 #? Combines multiple one-hot coding vectors

            if self.mode == ModulatorMode.PROBABILISTIC:
                mu_final, log_var_final = self.modulator(
                    c,
                    mu=mu_obs,
                    log_var=log_var_obs,
                    x_ctrl=context_x
                )
                z_final = self.reparameterize(mu_final, log_var_final)
            else: #? ModulatorMode.DETERMINISTIC
                z_final = self.modulator(
                    c,
                    z=z_obs,
                    x_ctrl=context_x
                )
                #? For loss calculation, use the original distribution
                mu_final, log_var_final = None, None

        #? No intervention is given, use the original distribution
        else:
            raise ValueError("c1 is mandatory for intervention!")

        #? --- 3. Shared Causal Transform and Decoding ---
        if self.is_causal and use_causal_layer:
            #? Normalize inputs to the DAGMA layer for stability ---
            z_initial_norm = self.dagma_norm(z_obs)
            z_final_norm = self.dagma_norm(z_final)

            u_interv = self.dagma_layer(z_final_norm)
            u_recon = self.dagma_layer(z_initial_norm)
            graph = self.dagma_layer.weight
        else:
            u_interv, u_recon, graph = z_final, z_obs, None

        #? --- 4. Decode for reconstruction ---
        y_hat = self.decode(u_interv)
        y_hat = self.clip_layer(y_hat)
        x_recon = self.decode(u_recon)
        x_recon = self.clip_layer(x_recon)

        #? --- 5. Format and return output ---
        outputs = CRLForwardPassOutput(
            y_hat=y_hat,
            x_recon=x_recon,
            z=z_final,
            z_obs=z_obs,
            G=graph,
            mu=mu_final,
            log_var=log_var_final,
            mu_obs=mu_obs,
            log_var_obs=log_var_obs,
        )

        if not return_dict:
            return list(outputs.values())

        return outputs
