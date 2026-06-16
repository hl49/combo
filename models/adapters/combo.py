import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

from ..backbones.backbones_base import BaseVisionTransformerBackbone
from .adapter_base import BaseAdapter

class MaskedConv2d(torch.nn.Module):
    """
    Conv2d with hierarchical gating:
      - model-level gates (one per input channel / model)
      - channel-level gates (fine-grained)
    """

    def __init__(
            self,
            conv: torch.nn.Conv2d, 
            activation_gates: str = "sigmoid",
            model_level: bool = False,
            num_models: int = 4,
            tau: float = 1.0,
            hardsoft_gates: bool = False,
            hard_threshold: float = 0.0,
            greedy_pruning: bool = False,
            min_alive: int = 1,
            group_pruning: bool = False,
        ):
        super().__init__()
        self.model_level = model_level
        self.num_models = num_models
        self.conv = conv
        self.ch_per_model = conv.in_channels // num_models if model_level else conv.in_channels
        in_channels = self.num_models if self.model_level else conv.in_channels
        self.tau = tau
        self.hardsoft_gates = hardsoft_gates
        self.hard_threshold = hard_threshold
        self.greedy_pruning = greedy_pruning
        self.min_alive = min_alive
        self.group_pruning = group_pruning
        # if group_pruning and not model_level:
        #     self.ch_per_model = conv.in_channels // num_models
            
        if activation_gates not in ["sigmoid", "relu"]:
            raise ValueError(f"Unsupported activation_gates: {activation_gates}")
        
        if activation_gates == "sigmoid":
            self.gate_logits = nn.Parameter(torch.zeros(in_channels))
            self.activation = nn.Sigmoid()
        elif activation_gates == "relu":
            self.gate_logits = nn.Parameter(torch.ones(in_channels))
            self.activation = nn.ReLU()
        
        self.register_buffer('keep_indices', torch.arange(in_channels).long())
    
    def set_keep_indices(self, keep_ch_indices: torch.Tensor) -> None:
        """Called by the pruning hook to permanently filter input channels.
        
        Args:
            keep_ch_indices: 1D tensor of channel indices to keep from the
                             full [B, 48, H, W] input.
        """
        self.register_buffer(
            'keep_indices',
            keep_ch_indices.clone().long()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, N, D]
        # x = x * self.gates.view(1, -1, 1, 1)  # Apply mask
        # step 1: filter input channels
        if self.keep_indices is not None:
            x = x[:, self.keep_indices, :, :] # [B, num_kept, N, D]
        # step 2: apply gates to remaining channels
        soft = self.activation(self.gate_logits / (self.tau + 1e-8))
        if not self.hardsoft_gates:
            gates = soft
            # Expand model-level gates to channel-level
            if self.model_level:
                gates = gates.repeat_interleave(self.ch_per_model)
        else:
            # soft = self.activation(self.gate_logits/(self.tau + 1e-8))
            hard = (soft > self.hard_threshold).float()
            # Keep min_alive to avoid setting all channels to 0
            if self.group_pruning:
                for i in range(self.num_models):
                    g_ini = i*self.ch_per_model
                    g_end = g_ini + self.ch_per_model
                    group_soft = soft[g_ini:g_end]
                    topk_idx = group_soft.detach().topk(self.min_alive).indices + g_ini
                    # n_survive = int(hard[g_ini:g_end].sum().item())
                    # if n_survive < self.min_alive:
                    hard[topk_idx] = 1.0
                # print(f"hard gates: {hard}")
            else:
                topk_idx = soft.detach().topk(self.min_alive).indices
                if hard.detach().sum() < self.min_alive:
                    hard[topk_idx] = 1.0
            if self.greedy_pruning:
                # gates = hard - soft.detach() + soft
                # gates = hard * soft 
                gates = hard.detach() * soft
            else:
                # gates = hard * soft 
                # gates = hard * soft + soft - soft.detach()
                gates = (hard * soft).detach() + soft - soft.detach()

        x = x * gates.view(1, -1, 1, 1)  # Apply mask
        return self.conv(x)


class ComBoAdapter(BaseAdapter):
    """
    Implementation of the ComBo adapter.
    """

    def __init__(
        self,
        backbone: BaseVisionTransformerBackbone,
        num_classes: int,
        freeze_backbone: bool = True,
        # Transformer parameters
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        token_pooling: str = "cls",  # Options: "cls", "avg", "map".
        patch_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_drop_rate: float = 0.0,
        rescale_after_layer_drop: bool = True,
        layers_to_keep: Optional[List[int]] = None,
        # Gating parameters
        sigmoid_gates: bool = False,
        **kwargs,
    ):
        self.feat_transformer_embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.token_pooling = token_pooling
        self.patch_drop_rate = patch_drop_rate
        self.drop_path_rate = drop_path_rate
        self.layer_drop_rate = layer_drop_rate
        self.rescale_after_layer_drop = rescale_after_layer_drop
        self.layers_to_keep = layers_to_keep
        self.sigmoid_gates = sigmoid_gates

        super().__init__(backbone, num_classes, freeze_backbone, **kwargs)

    def _setup_adapter(self, **kwargs):
        """Set up the ComBo adapter.
        Note: The linear projection described in the paper (Lambda) is implemented by the patch_embed that is within the timm VisionTransformer.
        By setting the img_size to [seq_len, embed_dim], patch_size to [1, embed_dim] and in_chans to num_layers, the patch embedding
        will project the stacked input features to the desired embedding dimension.
        """
        # Get sequence length directly from backbone token count
        seq_len = self.backbone.get_num_total_tokens()

        # Determine the number of input channels based on layers_to_keep
        if self.layers_to_keep is not None:
            in_chans = len(self.layers_to_keep)
        else:
            in_chans = self.backbone_num_layers

        # Create the Feature Transformer using timm's VisionTransformer
        pool_type = "token" if self.token_pooling == "cls" else self.token_pooling

        self.feat_transformer = VisionTransformer(
            img_size=[seq_len, self.backbone_embedding_dim],
            patch_size=[1, self.backbone_embedding_dim],
            in_chans=in_chans,
            num_classes=self.num_classes,
            embed_dim=self.feat_transformer_embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            global_pool=pool_type,
            class_token=self.token_pooling == "cls",
            pos_embed="learn",
            no_embed_class=False,
            reg_tokens=0,
            patch_drop_rate=self.patch_drop_rate,
            drop_path_rate=self.drop_path_rate,
            norm_layer=torch.nn.LayerNorm,
        )

        if self.sigmoid_gates:
            patch_embedding = self.feat_transformer.patch_embed
            original_conv = self.feat_transformer.patch_embed.proj
            patch_embedding.proj = MaskedConv2d(original_conv)

    def get_adapter_parameters(self) -> Dict[str, Any]:
        """Get parameters specific to this adapter for serialization."""
        return {
            "transformer_embed_dim": self.feat_transformer_embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
            "class_token": self.token_pooling == "cls",
            "token_pooling": self.token_pooling,
            "layers_to_keep": self.layers_to_keep,
            "layer_drop_rate": self.layer_drop_rate,
            "patch_drop_rate": self.patch_drop_rate,
            "drop_path_rate": self.drop_path_rate,
            "rescale_after_layer_drop": self.rescale_after_layer_drop,
        }

    def _apply_layer_dropout(
        self, layer_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Apply layer dropout during training by zeroing out selected layers. This was experimental and not used in the final paper."""
        if not self.training or self.layer_drop_rate <= 0 or len(layer_features) <= 1:
            return layer_features

        num_layers = len(layer_features)
        min_layers = 1

        # Create dropout mask - 1 means keep, 0 means drop
        keep_probs = torch.ones(num_layers, device=layer_features[0].device) * (
            1 - self.layer_drop_rate
        )
        mask = torch.bernoulli(keep_probs).bool()

        # Ensure we keep at least min_layers
        if mask.sum() < min_layers:
            indices = list(range(num_layers))
            random.shuffle(indices)
            for idx in indices[:min_layers]:
                mask[idx] = True

        # Zero out dropped layers instead of removing them
        processed_features = []
        for i, feat in enumerate(layer_features):
            if mask[i]:
                processed_features.append(feat)
            else:
                processed_features.append(torch.zeros_like(feat))

        # Rescale remaining features (inverted dropout)
        if (
            self.rescale_after_layer_drop
            and self.training
            and self.layer_drop_rate > 0
            and mask.sum() > 0
        ):
            scale = 1.0 / (1.0 - self.layer_drop_rate)
            processed_features = [f * scale for f in processed_features]

        return processed_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ComBo adapter."""
        # Extract feature maps from all layers across models
        # (Interpolation and normalisation happens inside get_features if configured)
        layer_features = self.backbone.get_features(x)

        # Filter layers if layers_to_keep is specified
        if self.layers_to_keep is not None:
            layer_features = [layer_features[i] for i in self.layers_to_keep]

        # Apply layer dropout during training (experimental, not used for the paper results)
        if self.training and self.layer_drop_rate > 0:
            layer_features = self._apply_layer_dropout(layer_features)

        # Stack features across layer dimension
        layer_features = torch.stack(layer_features, dim=1)

        # Process through the transformer (the patch embedding projects the stacked features to the desired embedding dimension, and the result token-wise embeddings are processed through the transformer blocks.)
        logits = self.feat_transformer(layer_features)

        return logits

    def get_regularization_losses(
        self,
        l1_scale: float = 0.0,
        l2_scale: float = 0.0,
        gl_scale: float = 0.0,
        struct_scale: float = 0.0,
        model_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate regularization losses for the adapter."""
        l1_loss = 0.0
        l2_loss = 0.0
        gl_loss = 0.0
        struct_loss = 0.0
        model_loss = 0.0

        patch_weight = self.feat_transformer.patch_embed.proj.weight

        if l1_scale > 0:
            l1_loss = l1_scale * torch.sum(torch.abs(patch_weight))

        if l2_scale > 0:
            l2_loss = l2_scale * torch.sum(patch_weight**2)

        if gl_scale > 0:
            embed_dim_norms = torch.sqrt(torch.sum(patch_weight**2, dim=(0, 2)) + 1e-8)
            gl_loss = gl_scale * torch.sum(embed_dim_norms)

        if struct_scale > 0:
            layer_norms = torch.sqrt(torch.sum(patch_weight**2, dim=(0, 2, 3)) + 1e-8)
            struct_loss = struct_scale * torch.sum(layer_norms)

        individual_model_norms = self._compute_individual_model_norms()
        if model_scale > 0 and individual_model_norms:
            model_norms_tensor = torch.stack(list(individual_model_norms.values()))
            model_loss = model_scale * torch.sum(model_norms_tensor)

        return l1_loss, l2_loss, gl_loss, struct_loss, model_loss

    def _compute_individual_model_norms(self) -> Dict[str, torch.Tensor]:
        """Compute individual model norms from patch embedding weights."""
        model_norms = {}
        patch_weight = self.feat_transformer.patch_embed.proj.weight

        if self.layers_to_keep is not None:
            layer_indices = self.layers_to_keep
        else:
            layer_indices = list(range(self.backbone_num_layers))

        backbone_layer_groups = {}

        if hasattr(self.backbone, "get_layer_to_model_mapping"):
            layer_to_model = self.backbone.get_layer_to_model_mapping()
            for i, layer_idx in enumerate(layer_indices):
                model_id = layer_to_model.get(layer_idx, 0)
                if model_id not in backbone_layer_groups:
                    backbone_layer_groups[model_id] = []
                backbone_layer_groups[model_id].append(i)
        else:
            backbone_layer_groups[0] = list(range(len(layer_indices)))

        for model_id, channel_indices in backbone_layer_groups.items():
            if channel_indices:
                model_patch_weights = patch_weight[:, channel_indices, :, :]
                model_norm = torch.sqrt(torch.sum(model_patch_weights**2) + 1e-8)

                if hasattr(self.backbone, "backbone_names"):
                    model_name = self.backbone.backbone_names[model_id]
                else:
                    model_name = f"model_{model_id}"

                model_norms[model_name] = model_norm

        return model_norms
