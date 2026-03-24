from typing import Any, Dict

import torch
import torch.nn as nn
from timm.layers import AttentionPoolLatent

from ..backbones.backbones_base import BaseVisionTransformerBackbone
from .adapter_base import BaseAdapter


class LinearProbeAdapter(BaseAdapter):
    """
    Linear Probe Adapter to train only a linear head on top of frozen features.

    This adapter extracts features from a specified layer of a backbone and
    applies a linear classifier to perform the task, keeping the backbone frozen.
    """

    def __init__(
        self,
        backbone: BaseVisionTransformerBackbone,
        num_classes: int,
        freeze_backbone: bool = True,
        layer_index: int = -1,  # Default to last layer
        token_pooling: str = "cls",  # Options: 'cls', 'cls_average', 'cls_concat', 'avg', 'map_all_tokens', 'map_spatial_tokens'
        map_num_heads: int = 12,
        dropout_rate: float = 0.0,
        **kwargs,
    ):
        self.layer_index = layer_index
        self.token_pooling = token_pooling
        self.map_num_heads = map_num_heads
        self.dropout_rate = dropout_rate

        super().__init__(backbone, num_classes, freeze_backbone, **kwargs)

    def _setup_adapter(self, **kwargs):
        """Set up linear probe adapter components."""
        # Remove the head of the original backbone
        if hasattr(self.backbone._model, "head"):
            self.backbone._model.head = nn.Identity()

        # Validate layer_index
        max_layer = self.backbone_num_layers - 1
        if self.layer_index < 0:
            # Convert negative indexing to positive
            self.layer_index = max(0, max_layer + self.layer_index + 1)
        elif self.layer_index > max_layer:
            # Clamp to maximum available layer
            self.layer_index = max_layer

        # Setup AttentionPoolLatent for map pooling
        if (
            self.token_pooling == "map_all_tokens"
            or self.token_pooling == "map_spatial_tokens"
        ):
            self.map_pooling = AttentionPoolLatent(
                self.backbone_embedding_dim, num_heads=self.map_num_heads
            )

        # Set up classifier components
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = nn.Identity()

        # Determine classifier input dimension based on token_pooling
        classifier_input_dim = self.backbone_embedding_dim
        if (
            self.token_pooling == "cls_concat"
            and self.backbone.get_num_cls_tokens() > 1
        ):
            classifier_input_dim = (
                self.backbone_embedding_dim * self.backbone.get_num_cls_tokens()
            )

        # Linear classification head
        self.classifier = nn.Linear(classifier_input_dim, self.num_classes)

    def extract_features(self, unified_tokens):
        """
        Extract features from unified token representation based on token_pooling strategy.
        """
        batch_size = unified_tokens.shape[0]
        num_cls = self.backbone.get_num_cls_tokens()

        if self.token_pooling == "map_all_tokens":
            features = self.map_pooling(unified_tokens)
        elif self.token_pooling == "map_spatial_tokens":
            spatial_tokens = unified_tokens[:, num_cls:, :]
            features = self.map_pooling(spatial_tokens)
        elif self.token_pooling.startswith("cls"):
            if num_cls == 0:
                # No CLS tokens available, fall back to avg pooling
                features = torch.mean(unified_tokens, dim=1)
            else:
                # Extract CLS tokens
                cls_tokens = unified_tokens[:, :num_cls, :]

                if self.token_pooling == "cls_average" and cls_tokens.shape[1] > 1:
                    # Average multiple CLS tokens
                    features = torch.mean(cls_tokens, dim=1)
                elif self.token_pooling == "cls_concat" and cls_tokens.shape[1] > 1:
                    # Concatenate multiple CLS tokens
                    features = cls_tokens.view(batch_size, -1)
                else:
                    # Use first CLS token or only one exists
                    features = cls_tokens[:, 0, :]
        else:  # "avg" pooling
            # Use spatial tokens (all tokens after CLS tokens)
            if num_cls > 0:
                spatial_tokens = unified_tokens[:, num_cls:, :]
                features = torch.mean(spatial_tokens, dim=1)
            else:
                features = torch.mean(unified_tokens, dim=1)

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the linear probing adapter."""
        # Get features from the specified layer index
        # Using the new get_features() method
        all_features = self.backbone.get_features(x)
        unified_tokens = all_features[self.layer_index]

        # Extract features based on pooling strategy
        features = self.extract_features(unified_tokens)

        # Apply dropout and classification head
        features = self.dropout(features)
        return self.classifier(features)

    def get_adapter_parameters(self) -> Dict[str, Any]:
        """Get parameters specific to this adapter for serialization."""
        params = {
            "layer_index": self.layer_index,
            "token_pooling": self.token_pooling,
            "dropout_rate": self.dropout_rate,
        }

        if (
            self.token_pooling == "map_all_tokens"
            or self.token_pooling == "map_spatial_tokens"
        ):
            params["map_num_heads"] = self.map_num_heads

        return params

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
