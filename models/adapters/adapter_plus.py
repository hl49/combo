from typing import Any, Dict, Literal, Tuple

import adapter_plus
import torch
import torch.nn as nn
from adapter_plus import Adapter
from omegaconf import OmegaConf
from timm.layers import AttentionPoolLatent, trunc_normal_
from timm.models.vision_transformer import Block, ResPostBlock
from timm.models.vision_transformer_sam import Block as SAMBlock

from ..backbones.backbones_base import BaseVisionTransformerBackbone
from .adapter_base import BaseAdapter


class PostAdapterBlockWrapper(nn.Module):
    """
    Wrapper for the AdapterBlock to handle the post-processing of the output.
    """

    def __init__(
        self,
        block: Block | ResPostBlock | SAMBlock,
        dim,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        adapter_config=None,
    ):
        super().__init__()
        assert adapter_config.config == "post", (
            "Adapter config must be 'post' for this wrapper"
        )
        self.block = block
        self.adapter = Adapter(
            dim,
            bottleneck_dim=adapter_config.dim,
            dropout=adapter_config.dropout,
            drop_path=adapter_config.drop_path,
            act_layer=act_layer if adapter_config.act_layer else None,
            norm_layer=norm_layer if adapter_config.norm_layer else None,
            bias=adapter_config.bias,
            scaling=adapter_config.scaling,
            init=adapter_config.init,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the adapter block."""
        x = self.block(x)
        if len(x.shape) == 3:
            x = self.adapter(x, skip=x)
        elif len(x.shape) == 4:  # case for SAM
            B, H, W, _ = x.shape
            x = x.reshape(B, H * W, -1)
            x = self.adapter(x, skip=x)
            x = x.reshape(B, H, W, -1)
        return x


class AdapterPlusAdapter(BaseAdapter):
    """
    AdapterPlus Adapter implementation (https://arxiv.org/abs/2406.06820v1).
    """

    def __init__(
        self,
        backbone: BaseVisionTransformerBackbone,
        num_classes: int,
        freeze_backbone: bool = True,
        config: str = "post",
        dim: int = 8,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        token_pooling: Literal[
            "cls",
            "cls_average",
            "cls_concat",
            "avg",
            "map_all_tokens",
            "map_spatial_tokens",
        ] = "cls",
        **kwargs,
    ):
        self.config = config
        self.dim = dim
        self.dropout = dropout
        self.drop_path = drop_path
        self.token_pooling = token_pooling

        super().__init__(backbone, num_classes, freeze_backbone, **kwargs)

    def _setup_adapter(self, **kwargs):
        """Set up adapter using the adapter_plus library."""
        adapterplus_conf = OmegaConf.create(
            f"""
            config: {self.config}
            act_layer: true
            norm_layer: false
            bias: true
            init: houlsby
            scaling: channel
            dim: {self.dim}
            attn_adapter: false
            dropout: {self.dropout}
            drop_path: {self.drop_path}
            """
        )

        # Look for blocks or residual blocks in the backbone
        for i, block in enumerate(self.backbone._model.blocks):
            if (
                isinstance(block, SAMBlock)
                or isinstance(block, Block)
                or isinstance(block, ResPostBlock)
            ):
                self.backbone._model.blocks[i] = PostAdapterBlockWrapper(
                    block,
                    dim=self.backbone_embedding_dim,  # Dummy values, will be overwritten
                    adapter_config=adapterplus_conf,
                )

        # Set parameter training requirements
        self.backbone.requires_grad_(False)  # Freeze most parameters
        for m in self.backbone._model.modules():
            if isinstance(m, adapter_plus.Adapter):
                m.requires_grad_(True)  # Unfreeze all adapter modules

        # Determine classifier input dimension based on token_pooling
        classifier_input_dim = self.backbone_embedding_dim
        if self.token_pooling == "cls_concat" and self.backbone_num_cls_tokens > 1:
            classifier_input_dim = (
                self.backbone_embedding_dim * self.backbone_num_cls_tokens
            )

        # Classification head to process the output features
        self.classifier = nn.Linear(classifier_input_dim, self.num_classes)
        trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)

        # Setup AttentionPoolLatent for "map" token pooling
        if (
            self.token_pooling == "map_all_tokens"
            or self.token_pooling == "map_spatial_tokens"
        ):
            self.map_pooling = AttentionPoolLatent(
                self.backbone_embedding_dim, num_heads=self.map_num_heads
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with AdapterPlus adaptation."""
        # Process input
        all_tokens, cls_tokens, spatial_tokens = self.backbone(x)

        # Extract features based on token_pooling strategy
        if self.token_pooling.startswith("cls"):
            features = cls_tokens
            if features.shape[1] > 1:  # Multiple CLS tokens
                if self.token_pooling == "cls_average":
                    features = torch.mean(features, dim=1)  # Average CLS tokens
                elif self.token_pooling == "cls_concat":
                    batch_size = features.shape[0]
                    features = features.view(batch_size, -1)  # Flatten all CLS tokens
                else:
                    # Default to first token if not specified
                    features = features[:, 0]
            else:
                features = features.squeeze(1)  # Single CLS token
        elif self.token_pooling == "map_all_tokens":
            features = self.map_pooling(all_tokens)  # [B, C]
        elif self.token_pooling == "map_spatial_tokens":
            features = self.map_pooling(spatial_tokens)
        else:  # "avg" - Average pooling over spatial tokens
            features = torch.mean(spatial_tokens, dim=1)

        # Apply classification head
        return self.classifier(features)

    def get_adapter_parameters(self) -> Dict[str, Any]:
        """Get parameters specific to this adapter for serialization."""
        params = {
            "config": self.config,
            "dim": self.dim,
            "dropout": self.dropout,
            "drop_path": self.drop_path,
        }

        if hasattr(self, "modules_to_save") and self.modules_to_save:
            params["modules_to_save"] = self.modules_to_save

        return params

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
