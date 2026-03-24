from typing import Any, Dict, List, Literal, Optional, Set, Union

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from timm.layers import AttentionPoolLatent, PatchDropout

from ..backbones.backbones_base import BaseVisionTransformerBackbone
from .adapter_base import BaseAdapter


class LoRAAdapter(BaseAdapter):
    """
    LoRA (Low-Rank Adaptation) Adapter implementation using the PEFT library.
    """

    def __init__(
        self,
        backbone: BaseVisionTransformerBackbone,
        num_classes: int,
        freeze_backbone: bool = True,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.1,
        target_modules: Optional[Union[List[str], Set[str]]] = None,
        token_pooling: Literal[
            "cls",
            "cls_average",
            "cls_concat",
            "avg",
            "map_all_tokens",
            "map_spatial_tokens",
        ] = "cls",
        modules_to_save: Optional[List[str]] = None,
        map_num_heads: int = 12,
        patch_drop_rate: float = 0.0,
        **kwargs,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.token_pooling = token_pooling
        self.modules_to_save = modules_to_save
        self.map_num_heads = map_num_heads
        self.patch_drop_rate = patch_drop_rate

        super().__init__(backbone, num_classes, freeze_backbone, **kwargs)

    def _setup_adapter(self, **kwargs):
        """Set up LoRA adapter using PEFT library."""
        # Add check for CombinedBackbone
        if (
            hasattr(self.backbone, "backbones")
            and len(getattr(self.backbone, "backbones", [])) > 1
        ):
            raise ValueError(
                "LoRA adapter does not support CombinedBackbone with multiple backbones. "
                "Please use a single backbone with LoRA."
            )

        # Default target modules for ViT if not specified
        if self.target_modules is None:
            self.target_modules = ["query", "key", "value", "out"]

        # If target_modules is provided as a single string in the config
        # Convert it to a list
        if isinstance(self.target_modules, str):
            if "," in self.target_modules:
                self.target_modules = [
                    module.strip() for module in self.target_modules.split(",")
                ]
            else:
                self.target_modules = [self.target_modules]

        # Convert to list if it's a set
        if isinstance(self.target_modules, set):
            self.target_modules = list(self.target_modules)

        # Process modules_to_save similarly
        if isinstance(self.modules_to_save, str):
            if "," in self.modules_to_save:
                self.modules_to_save = [
                    module.strip() for module in self.modules_to_save.split(",")
                ]
            else:
                self.modules_to_save = [self.modules_to_save]

        # Configure LoRA
        lora_config_args = {
            "r": self.rank,
            "lora_alpha": self.alpha,
            "target_modules": self.target_modules,
            "lora_dropout": self.dropout,
        }

        # Only add modules_to_save if it's not None
        if self.modules_to_save:
            lora_config_args["modules_to_save"] = self.modules_to_save

        # Process any additional kwargs that might be in the config
        for key, value in kwargs.items():
            if key not in lora_config_args and not hasattr(self, key):
                setattr(self, key, value)

        self.lora_config = LoraConfig(**lora_config_args)

        # Potentially replace the patch drop in the backbone
        if self.patch_drop_rate > 0.0:
            if hasattr(self.backbone._model, "num_prefix_tokens"):
                num_prefix_tokens = self.backbone._model.num_prefix_tokens
            else:
                num_prefix_tokens = self.backbone.actual_num_cls_tokens
            # This is the case for the timm ViT models
            if hasattr(self.backbone._model, "patch_drop"):
                self.backbone._model.patch_drop = PatchDropout(
                    self.patch_drop_rate, num_prefix_tokens=num_prefix_tokens
                )
            # This is the case for the Radio model
            elif hasattr(self.backbone, "patch_drop"):
                self.backbone.patch_drop = PatchDropout(
                    self.patch_drop_rate, num_prefix_tokens=num_prefix_tokens
                )
            else:
                raise ValueError("PatchDropout is not available in the backbone model.")

        # Get PEFT model - apply LoRA to the backbone's underlying model
        self.lora_model = get_peft_model(self.backbone, self.lora_config)

        # Determine classifier input dimension based on token_pooling
        classifier_input_dim = self.backbone_embedding_dim
        if self.token_pooling == "cls_concat" and self.backbone_num_cls_tokens > 1:
            classifier_input_dim = (
                self.backbone_embedding_dim * self.backbone_num_cls_tokens
            )

        # Classification head to process the output features
        self.classifier = nn.Linear(classifier_input_dim, self.num_classes)

        # Setup AttentionPoolLatent for "map" token pooling
        if (
            self.token_pooling == "map_all_tokens"
            or self.token_pooling == "map_spatial_tokens"
        ):
            self.map_pooling = AttentionPoolLatent(
                self.backbone_embedding_dim, num_heads=self.map_num_heads
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with a LoRA adapter."""
        # Forward pass through adapted model
        all_tokens, cls_tokens, spatial_tokens = self.lora_model(x)

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
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "token_pooling": self.token_pooling,
        }

        if hasattr(self, "modules_to_save") and self.modules_to_save:
            params["modules_to_save"] = self.modules_to_save

        return params

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
