"""
Combined Backbones implementation.

Provides a wrapper to combine multiple backbone models into a single
logical model for feature extraction.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .backbones_base import BaseVisionTransformerBackbone


class CombinedBackbone(nn.Module):
    """
    A backbone that combines multiple backbone models into a single logical model
    with standardized embedding dimensions and token counts.
    """

    def __init__(
        self,
        backbones: List[BaseVisionTransformerBackbone],
        backbone_names: Optional[List[str]] = None,
        name: Optional[str] = None,
        target_embed_dim: Optional[int] = None,
        cls_token_strategy: str = "max_pad",  # 'max_pad', 'all', 'first_only', 'spatial_only'
        **kwargs,
    ):
        """
        Initialise with a list of backbone models.

        Args:
            backbones: List of backbone models to combine
            backbone_names: Optional list of names for each backbone
            name: Name for the combined backbone
            target_embed_dim: Target embedding dimension for all outputs
                             (if None, uses maximum dimension from all backbones)
            cls_token_strategy: Strategy for handling cls tokens from multiple backbones
                - 'max_pad': Use the maximum number of cls tokens, padding with zeros
                - 'all': Concatenate all cls tokens from all backbones
                - 'first_only': Use only the first cls token from each backbone
                - 'spatial_only': Discard all cls tokens, use only spatial features
            **kwargs: Additional arguments (for compatibility with BaseVisionTransformerBackbone)
        """
        super().__init__()

        if not backbones:
            raise ValueError("Must provide at least one backbone")

        if backbone_names and len(backbone_names) != len(backbones):
            raise ValueError("backbone_names must match the number of backbones")

        valid_strategies = ["max_pad", "all", "first_only", "spatial_only"]
        if cls_token_strategy not in valid_strategies:
            raise ValueError(f"cls_token_strategy must be one of {valid_strategies}")

        # Generate name if not provided
        if name is None:
            name = "combined_" + "_".join([b.get_name() for b in backbones])

        # Get target embedding dimension (max by default)
        self.original_embed_dims = [
            backbone.get_embedding_dim() for backbone in backbones
        ]
        if target_embed_dim is None:
            target_embed_dim = max(self.original_embed_dims)

        # Extract cls token strategy
        self.cls_token_strategy = cls_token_strategy

        # Calculate effective number of cls tokens based on strategy
        if self.cls_token_strategy == "spatial_only":
            self.num_cls_tokens = 0
        elif self.cls_token_strategy == "first_only":
            self.num_cls_tokens = 1
        elif self.cls_token_strategy == "all":
            self.num_cls_tokens = sum([b.get_num_cls_tokens() for b in backbones])
        else:  # 'max_pad'
            self.num_cls_tokens = max([b.get_num_cls_tokens() for b in backbones])

        # Save parameters
        self.name = name
        self.target_embed_dim = target_embed_dim
        self.backbones = nn.ModuleList(backbones)
        self.backbone_names = backbone_names or [
            f"backbone_{i}" for i in range(len(backbones))
        ]
        self.embed_dim = target_embed_dim

        # Extract useful kwargs
        self.use_layernorm = kwargs.get("use_layernorm", False)
        self.use_norm2_outputs = kwargs.get("use_norm2_outputs", False)

        # Create projection layers if needed
        self.projections = nn.ModuleList()
        for dim in self.original_embed_dims:
            if dim != target_embed_dim:
                self.projections.append(nn.Linear(dim, target_embed_dim))
            else:
                self.projections.append(nn.Identity())

        # Count total number of layers
        self.num_blocks = sum([backbone.get_num_layers() for backbone in backbones])

    def _collect_backbone_features(
        self, x: torch.Tensor
    ) -> List[Tuple[List[torch.Tensor], int]]:
        """Collect features from all backbones."""
        backbone_features_list = []
        for i, backbone in enumerate(self.backbones):
            backbone_features = backbone.get_features(x)
            backbone_features_list.append((backbone_features, i))
        return backbone_features_list

    def _process_spatial_only(
        self, backbone_features_list: List[Tuple[List[torch.Tensor], int]]
    ) -> List[torch.Tensor]:
        """Process features using the 'spatial_only' strategy."""
        all_features = []

        for backbone_features, i in backbone_features_list:
            projection = self.projections[i]
            for feat in backbone_features:
                num_cls = self.backbones[i].get_num_cls_tokens()
                # Extract spatial tokens
                spatial_tokens = feat[:, num_cls:, :]
                # Apply projection if needed
                spatial_tokens = projection(spatial_tokens)
                all_features.append(spatial_tokens)

        return all_features

    def _process_first_only(
        self, backbone_features_list: List[Tuple[List[torch.Tensor], int]]
    ) -> List[torch.Tensor]:
        """Process features using the 'first_only' strategy."""
        all_features = []

        for backbone_features, i in backbone_features_list:
            projection = self.projections[i]
            for feat in backbone_features:
                batch_size, seq_len, dim = feat.shape
                num_cls = self.backbones[i].get_num_cls_tokens()

                if num_cls > 0:
                    cls_token = feat[:, 0:1, :]  # Just the first token
                    spatial_tokens = feat[:, num_cls:, :]
                else:
                    # No cls tokens, create dummy
                    cls_token = torch.zeros((batch_size, 1, dim), device=feat.device)
                    spatial_tokens = feat

                # Apply projection if needed
                cls_token = projection(cls_token)
                spatial_tokens = projection(spatial_tokens)

                # Combine and add to features
                combined = torch.cat([cls_token, spatial_tokens], dim=1)
                all_features.append(combined)

        return all_features

    def _process_all_cls(
        self, backbone_features_list: List[Tuple[List[torch.Tensor], int]]
    ) -> List[torch.Tensor]:
        """Process features using the 'all' strategy."""
        all_features = []

        # First, determine the total number and position mapping for cls tokens
        cls_counts = [b.get_num_cls_tokens() for b in self.backbones]
        cls_positions = {}  # Maps backbone_idx -> (start_pos, end_pos)

        global_pos = 0
        for i, count in enumerate(cls_counts):
            cls_positions[i] = (global_pos, global_pos + count)
            global_pos += count

        # Now process each backbone's features
        for backbone_features, i in backbone_features_list:
            projection = self.projections[i]
            for feat in backbone_features:
                batch_size, seq_len, dim = feat.shape
                num_cls = cls_counts[i]

                # Create placeholder for all cls tokens
                all_cls_tokens = torch.zeros(
                    (batch_size, self.num_cls_tokens, self.target_embed_dim),
                    device=feat.device,
                    dtype=feat.dtype,
                )

                # Extract and project this backbone's cls tokens
                if num_cls > 0:
                    cls_tokens = feat[:, :num_cls, :]
                    cls_tokens = projection(cls_tokens)

                    # Place cls tokens in their designated positions
                    start_pos, end_pos = cls_positions[i]
                    all_cls_tokens[:, start_pos:end_pos, :] = cls_tokens

                # Extract and project this backbone's spatial tokens
                spatial_tokens = feat[:, num_cls:, :]
                spatial_tokens = projection(spatial_tokens)

                # Combine and add to features
                combined = torch.cat([all_cls_tokens, spatial_tokens], dim=1)
                all_features.append(combined)

        return all_features

    def _process_max_pad(
        self, backbone_features_list: List[Tuple[List[torch.Tensor], int]]
    ) -> List[torch.Tensor]:
        """Process features using the 'max_pad' strategy."""
        all_features = []

        for backbone_features, i in backbone_features_list:
            projection = self.projections[i]
            for feat in backbone_features:
                batch_size, seq_len, dim = feat.shape
                num_cls = self.backbones[i].get_num_cls_tokens()
                cls_tokens = feat[:, :num_cls, :]
                spatial_tokens = feat[:, num_cls:, :]

                # Apply projection
                cls_tokens = projection(cls_tokens)
                spatial_tokens = projection(spatial_tokens)

                proj_dim = self.target_embed_dim

                # Pad or truncate cls tokens
                if num_cls < self.num_cls_tokens:
                    padding = torch.zeros(
                        (batch_size, self.num_cls_tokens - num_cls, proj_dim),
                        device=feat.device,
                        dtype=feat.dtype,
                    )
                    cls_tokens = torch.cat([cls_tokens, padding], dim=1)
                else:
                    cls_tokens = cls_tokens[:, : self.num_cls_tokens, :]

                # Combine and add to features
                combined = torch.cat([cls_tokens, spatial_tokens], dim=1)
                all_features.append(combined)

        return all_features

    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get standardized features from all backbones.
        Handles cls tokens according to cls_token_strategy.
        """
        backbone_features_list = self._collect_backbone_features(x)

        # Dispatch to appropriate strategy implementation
        if self.cls_token_strategy == "spatial_only":
            return self._process_spatial_only(backbone_features_list)
        elif self.cls_token_strategy == "first_only":
            return self._process_first_only(backbone_features_list)
        elif self.cls_token_strategy == "all":
            return self._process_all_cls(backbone_features_list)
        else:  # 'max_pad'
            return self._process_max_pad(backbone_features_list)

    def forward(
        self, x: torch.Tensor, normalize_output: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through all backbones."""
        results = {}
        for name, backbone in zip(self.backbone_names, self.backbones):
            results[name] = backbone(x)
        return results

    # Interface methods for compatibility with BaseVisionTransformerBackbone

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of the backbone."""
        return self.embed_dim

    def get_num_cls_tokens(self) -> int:
        """Get the number of class tokens used by the backbone."""
        return self.num_cls_tokens

    def get_num_spatial_tokens(self) -> int:
        """
        Get the number of spatial tokens in this backbone.
        For combined backbones, this is based on the first backbone's spatial tokens
        since all backbones should process the same input dimensions.
        """
        return self.backbones[0].get_num_spatial_tokens()

    def get_num_total_tokens(self) -> int:
        """Get the total number of tokens (cls + spatial) in this backbone."""
        return self.get_num_cls_tokens() + self.get_num_spatial_tokens()

    def get_num_layers(self) -> int:
        """Get the number of transformer layers/blocks in the backbone."""
        return self.num_blocks

    def get_name(self) -> str:
        """Get the descriptive name of the backbone."""
        return self.name

    def freeze_all_backbones(self) -> None:
        """Freeze all parameters in all backbones."""
        for backbone in self.backbones:
            for param in backbone.parameters():
                param.requires_grad = False

    def unfreeze_all_backbones(self) -> None:
        """Unfreeze all parameters in all backbones."""
        for backbone in self.backbones:
            for param in backbone.parameters():
                param.requires_grad = True

    def get_layer_to_model_mapping(self) -> Dict[int, int]:
        """Get mapping from global layer index to backbone model index."""
        layer_to_model = {}
        global_layer_idx = 0

        for model_idx, backbone in enumerate(self.backbones):
            num_layers = backbone.get_num_layers()
            for local_layer_idx in range(num_layers):
                layer_to_model[global_layer_idx] = model_idx
                global_layer_idx += 1

        return layer_to_model
