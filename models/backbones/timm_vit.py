from typing import Dict, List, Optional, Tuple, Union

import einops
import timm
import torch
import torch.nn as nn

from .backbones_base import BaseVisionTransformerBackbone


class TimmVisionTransformerBackbone(BaseVisionTransformerBackbone):
    """
    A Vision Transformer backbone implementation using timm library.

    Focused on standard Vision Transformer models (ViT architecture).
    """

    def __init__(
        self,
        model_path: str = "vit_base_patch16_224",
        target_grid_size: Optional[Union[int, Tuple[int, int]]] = None,
        timm_kwargs: Optional[Dict] = None,
        name: Optional[str] = None,
        include_register_tokens: bool = False,
        drop_path_rate: float = 0.0,
        init_last_norm_from_pretrained: bool = False,
        **kwargs,
    ):
        """
        Initialise the Vision Transformer backbone.

        Args:
            model_path: Path/name of the Vision Transformer model to load from timm
            target_grid_size: Target size for feature grid (default: None)
            timm_kwargs: Additional arguments passed to timm.create_model
            name: Short descriptive name for the model
            include_register_tokens: Whether to include register tokens with CLS tokens
            drop_path_rate: Drop path rate for the model (default: 0.0)
            init_last_norm_from_pretrained: Whether to initialise the last normalisation layer from pretrained weights
            **kwargs: Additional arguments passed to the base class
        """
        self.model_path = model_path
        self.timm_kwargs = timm_kwargs or {}
        self.include_register_tokens = include_register_tokens
        self.drop_path_rate = drop_path_rate
        self.init_last_norm_from_pretrained = init_last_norm_from_pretrained

        # Generate a default name if not provided
        if name is None:
            name = model_path.split("/")[-1].split(".")[0]

        super().__init__(
            target_grid_size=target_grid_size,
            name=name,
            **kwargs,
        )

    def _setup_model(self, **kwargs):
        """Set up the Vision Transformer model using timm."""

        if "sam" in self.name:
            self._model = timm.create_model(
                self.model_path,
                pretrained=True,
                drop_path_rate=self.drop_path_rate,
                **self.timm_kwargs,
            )
        else:
            self._model = timm.create_model(
                self.model_path,
                pretrained=True,
                dynamic_img_size=True,
                drop_path_rate=self.drop_path_rate,
                **self.timm_kwargs,
            )

        # For standard ViT, directly access attributes we know exist
        self.patch_size = self._model.patch_embed.patch_size
        if isinstance(self.patch_size, (list, tuple)):
            self.patch_size = self.patch_size[0]

        # Store properties needed by adapters
        self.embed_dim = self._model.embed_dim
        self.num_blocks = len(self._model.blocks)
        if hasattr(self._model, "num_prefix_tokens"):
            self.num_prefix_tokens = self._model.num_prefix_tokens
            # This is a standard vision_transformer from timm
            if self.include_register_tokens:
                self.actual_num_cls_tokens = self.num_prefix_tokens
            else:
                self.actual_num_cls_tokens = 1 if self._model.has_class_token else 0
        else:
            # This would be for SAM, which doesn't have this attribute (see vision_transformer_sam.py from timm)
            self.actual_num_cls_tokens = 0
            self.num_prefix_tokens = 0

        # Get normalisation parameters from model config
        self.mean = torch.tensor(
            self._model.default_cfg.get("mean", [0.485, 0.456, 0.406])
        )
        self.std = torch.tensor(
            self._model.default_cfg.get("std", [0.229, 0.224, 0.225])
        )

        if self.init_last_norm_from_pretrained:
            self.norm = self._model.norm
        else:
            self.norm = nn.LayerNorm(self.embed_dim, eps=1e-6)

        # We handle the normalisation ourselves, so disable it in the model
        self._model.prune_intermediate_layers(prune_norm=True, prune_head=True)

    def get_actual_num_cls_tokens(self) -> int:
        """Get the actual number of cls tokens in this model."""
        return self.actual_num_cls_tokens

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply model-specific normalisation to input tensor."""
        # Move normalisation parameters to the same device as input
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)

        # Normalise the input tensor
        if x.ndim == 3:  # Single image: [C, H, W]
            x = x.unsqueeze(0)  # Add batch dimension

        # Apply normalisation: (x - mean) / std
        x = x.sub(self.mean[None, :, None, None]).div(self.std[None, :, None, None])

        return x

    def _extract_feature_map_and_cls_tokens(
        self, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract feature map and class tokens from the output tokens."""
        # Extract all prefix tokens, not just the first one
        cls_tokens = tokens[:, : self.actual_num_cls_tokens].detach()
        patch_tokens = tokens[:, self.num_prefix_tokens :].detach()

        # Calculate feature map dimensions (assuming square grid)
        H = W = int(patch_tokens.shape[1] ** 0.5)

        # Reshape patch tokens to feature map [B, D, H, W]
        feature_map = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0], self.embed_dim, H, W
        )

        # Transpose cls_tokens to have embedding dim at index 1
        cls_tokens = cls_tokens.transpose(1, 2)  # [B, D, num_cls_tokens]

        return feature_map, cls_tokens

    # def _get_raw_block_outputs(
    #     self, x: torch.Tensor
    # ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    #     """Get raw outputs from each transformer block."""
    #     x = self._normalize_input(x)
    #     has_prefix = self.num_prefix_tokens > 0

    #     with torch.no_grad():
            
    #         intermediates = self._model.forward_intermediates(
    #             x,
    #             indices=None,        # capture all blocks
    #             norm=False,          # raw pre-norm outputs
    #             stop_early=False,
    #             output_fmt='NCHW',    # keep (B, seq_len, dim) — no NCHW reshape
    #             intermediates_only=True,     # only return intermediates, not final output
    #             **({'return_prefix_tokens': True} if has_prefix else {}),  # include prefix tokens in output
    #         )
            
    #         outputs = []
    #         for intem in intermediates:
    #             if has_prefix:
    #                 intermediate, prefix = intem
    #             else:
    #                 intermediate, prefix = intem, None
    #             outputs.append((intermediate, prefix))

    #         return outputs

    def _get_raw_block_outputs(
        self, x: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get raw outputs from each transformer block."""
        x = self._normalize_input(x)

        with torch.no_grad():
            outputs = []
            hooks = []

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]

                # If output has 4 dimensions, it is from SAM with shape (B, H, W, D), and we need to reshape it to( B, H*W, D)
                if output.ndim == 4:
                    output = einops.rearrange(output, "b h w d -> b (h w) d")

                feature_map, cls_tokens = self._extract_feature_map_and_cls_tokens(
                    output
                )
                outputs.append((feature_map, cls_tokens))

            # Register hooks for each block
            for block in self._model.blocks:
                hooks.append(block.register_forward_hook(hook_fn))

            # Run forward pass
            _ = self._model(x)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            return outputs

    def _get_raw_norm2_outputs(
        self, x: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get outputs after each norm2 in transformer blocks."""
        x = self._normalize_input(x)

        with torch.no_grad():
            outputs = []
            hooks = []

            # For standard ViT, we know it's called norm2
            for i, block in enumerate(self._model.blocks):

                def hook_fn(module, input, output, i=i):
                    if isinstance(output, tuple):
                        output = output[0]
                    feature_map, cls_tokens = self._extract_feature_map_and_cls_tokens(
                        output
                    )
                    outputs.append((feature_map, cls_tokens))

                hooks.append(block.norm2.register_forward_hook(hook_fn))

            # Run forward pass
            _ = self._model(x)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            return outputs

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the backbone model.

        Args:
            x: Input tensor(s) of shape [B, C, H, W] or [C, H, W]

        Returns:
            Tuple containing:
                - all_tokens: Tensor of shape [B, num_cls_tokens + num_patches, embed_dim] representing the final layer's output
                - cls_tokens: Tensor of shape [B, num_cls_tokens, embed_dim] containing only the CLS tokens
                - spatial_tokens: Tensor of shape [B, num_patches, embed_dim] containing only the spatial tokens
        """
        # Always normalise input before passing to the model
        x = self._normalize_input(x)

        # Run the model
        tokens = self._model.forward_features(x)

        # Flatten the output if it's a 4D tensor
        if tokens.ndim == 4:
            tokens = einops.rearrange(tokens, "b d h w -> b (h w) d")

        tokens = self.norm(tokens)

        # Separate class tokens from spatial tokens
        cls_tokens = tokens[:, : self.actual_num_cls_tokens]  # Shape: [B, num_cls, D]
        spatial_tokens = tokens[
            :, self.num_prefix_tokens :
        ]  # Shape: [B, num_patches, D]

        # Return standardized output format
        return tokens, cls_tokens, spatial_tokens
