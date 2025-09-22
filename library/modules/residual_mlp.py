import typing as t
import enum
import torch
import torch.nn as nn
from .act_layer import get_act_layer
from .norm import get_norm_layer
from ..utils import core as my_utils
from ..consts import NormPlacement

class BasicResidualMLPBlock(nn.Module):
    """
    A basic residual MLP block with configurable normalization and activation.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        num_layers: int = 2,
        act_layer: str = 'relu',
        act_layer_kwargs: dict | None = None,
        norm_layer: str | None = None,
        norm_layer_kwargs: dict | None = None,
        norm_placement: str | NormPlacement = NormPlacement.ALL,
        dropout: float | None = None,
        use_residual: bool = False,
    ):
        super(BasicResidualMLPBlock, self).__init__()

        if hidden_dim is None:
            hidden_dim = in_dim
        if out_dim is None:
            out_dim = in_dim

        self.use_residual = use_residual

        #? Implement a projection shortcut for the residual connection.
        self.skip_connection = nn.Identity()
        if self.use_residual and in_dim != out_dim:
            self.skip_connection = nn.Linear(
                in_features=in_dim,
                out_features=out_dim,
            )

        act_layer_cls = get_act_layer(act_layer)
        act_layer_kwargs = my_utils.ensure_dict(act_layer_kwargs)

        norm_layer_cls = get_norm_layer(norm_layer)
        norm_layer_kwargs = my_utils.ensure_dict(norm_layer_kwargs)
        #? Simplified the conversion by using the StrEnum constructor directly.
        self.norm_placement = NormPlacement(norm_placement)

        #? Configure layers sequentially.
        self.layers = nn.Sequential()
        self.layers.add_module(
            'input_layer',
            nn.Linear(
                in_features=in_dim,
                out_features=hidden_dim,
            )
        )

        if norm_layer_cls and self.norm_placement in [NormPlacement.PRE, NormPlacement.ALL]:
            self.layers.add_module(
                'input_norm',
                norm_layer_cls(
                    hidden_dim,
                    **norm_layer_kwargs,
                )
            )

        self.layers.add_module(
            'input_act',
            act_layer_cls(**act_layer_kwargs),
        )

        if dropout is not None and dropout > 0:
            self.layers.add_module(
                'input_dropout',
                nn.Dropout(dropout),
            )

        #? Configure mid layers.
        for i in range(num_layers - 2):
            self.layers.add_module(
                f'layer_{i}',
                nn.Linear(
                    in_features=hidden_dim,
                    out_features=hidden_dim,
                )
            )

            if norm_layer_cls and self.norm_placement in [NormPlacement.MID, NormPlacement.ALL]:
                self.layers.add_module(
                    f'mid_norm_{i}',
                    norm_layer_cls(
                        hidden_dim,
                        **norm_layer_kwargs,
                    )
                )

            self.layers.add_module(
                f'layer_act_{i}',
                act_layer_cls(**act_layer_kwargs),
            )

            if dropout is not None and dropout > 0:
                self.layers.add_module(
                    f'dropout_{i}',
                    nn.Dropout(dropout),
                )

        #? Add the final linear layer without activation.
        self.layers.add_module(
            'output_layer',
            nn.Linear(
                in_features=hidden_dim,
                out_features=out_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the residual MLP block."""
        residual = x
        out = self.layers(x)

        if self.use_residual:
            out = out + self.skip_connection(residual)
            
        return out


class ResidualMLP(nn.Module):
    """
    A residual MLP with multiple blocks and configurable input/output layers.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        num_blocks: int = 3,
        in_act_layer: str = 'relu',
        in_act_layer_kwargs: dict | None = None,
        block_num_layers: int = 2,
        block_hidden_dim: int | None = None,
        block_act_layer: str = 'relu',
        block_norm_layer: str | None = None,
        block_norm_layer_kwargs: dict | None = None,
        block_norm_placement: str | NormPlacement = NormPlacement.ALL,
        block_dropout: float | None = None,
        out_block_ena_clip: bool = False,
        out_block_clip_kwargs: dict | None = None,
    ):
        super(ResidualMLP, self).__init__()
        
        in_act_cls = get_act_layer(in_act_layer)
        in_act_kwargs = my_utils.ensure_dict(in_act_layer_kwargs)

        if output_dim is None:
            output_dim = input_dim
        if hidden_dim is None:
            hidden_dim = input_dim
        if block_hidden_dim is None:
            block_hidden_dim = hidden_dim

        #_ Input block / Stem
        self.in_block = nn.Sequential(
            nn.Linear(
                in_features=input_dim,
                out_features=hidden_dim,
            ),
            in_act_cls(**in_act_kwargs),
        )

        #? Middle residual blocks
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            self.blocks.add_module(
                f'block_{i}', 
                BasicResidualMLPBlock(
                    in_dim=hidden_dim,
                    hidden_dim=block_hidden_dim,
                    out_dim=hidden_dim,
                    num_layers=block_num_layers,
                    act_layer=block_act_layer,
                    norm_layer=block_norm_layer,
                    norm_layer_kwargs=block_norm_layer_kwargs,
                    norm_placement=block_norm_placement,
                    dropout=block_dropout,
                    use_residual=True,
                )
            )

        #? Output block / Head
        self.out_block = nn.Sequential()
        self.out_block.add_module(
            'output_layer',
            nn.Linear(
                in_features=hidden_dim,
                out_features=output_dim,
            )
        )
        
        if out_block_ena_clip:
            out_block_clip_kwargs = my_utils.ensure_dict(out_block_clip_kwargs)
            self.out_block.add_module(
                'clip_layer',
                nn.Hardtanh(**out_block_clip_kwargs),
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the residual MLP model."""
        x = self.in_block(x)
        x = self.blocks(x)
        x = self.out_block(x)
        return x