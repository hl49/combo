import abc
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseVisionTransformerBackbone(nn.Module, abc.ABC):
    """
    Abstract base class for vision transformer backbones.

    Provides a common interface for various vision transformer models
    including SAM, DFN-CLIP, SigLIP, DINOv2, and ViT.
    """

    def __init__(
        self,
        target_grid_size: Optional[Union[int, Tuple[int, int]]] = None,
        cls_token_strategy: str = "all",
        num_cls_tokens: int = 1,
        use_layernorm: Union[bool, str] = False,
        name: Optional[str] = None,
        use_norm2_outputs: bool = False,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs,
    ):
        """
        Initialise the backbone.

        Args:
            target_grid_size: Target size for feature grid (int for square, tuple for rectangle)
            cls_token_strategy: Strategy for handling cls tokens ('all', 'first_only', 'spatial_only', 'fixed')
            num_cls_tokens: Number of cls tokens for 'fixed' strategy
            use_layernorm: Whether to apply non-learnable layernorm to features
            name: Optional name for the backbone
            use_norm2_outputs: Whether to use outputs after norm2 layer (True) or block outputs (False)
            image_size: Input image size (int for square, tuple for rectangle)
            **kwargs: Additional model-specific parameters
        """
        super().__init__()

        self.name = name
        self.use_norm2_outputs = use_norm2_outputs
        self.cls_token_strategy = cls_token_strategy
        self.num_cls_tokens = num_cls_tokens  # Used when strategy is 'fixed'

        # Handle layernorm options (keep backward compatibility)
        if isinstance(use_layernorm, bool):
            self.layernorm_type = "embed_only" if use_layernorm else None
        else:
            self.layernorm_type = use_layernorm

        self.use_layernorm = self.layernorm_type is not None

        # Handle image size
        if image_size is not None:
            if isinstance(image_size, int):
                self.image_size = (image_size, image_size)
            else:
                self.image_size = image_size
        else:
            self.image_size = (224, 224)  # Default size

        # Initialise token count tracking
        self.token_counts = {
            "cls": None,  # Number of cls tokens
            "spatial": None,  # Number of spatial tokens
            "total": None,  # Total number of tokens
        }

        # Handle target grid size
        if target_grid_size is not None:
            if isinstance(target_grid_size, int):
                self.target_grid_size = (target_grid_size, target_grid_size)
            else:
                self.target_grid_size = target_grid_size
        else:
            self.target_grid_size = None

        self._model = None
        self._setup_model(**kwargs)

        # Calculate token counts after model setup
        self._calculate_token_counts()

        # Create non-learnable layernorms (one per layer) if enabled
        self.layernorms = None
        if self.use_layernorm:
            # Will be initialised after model setup when num_blocks and embed_dim are available
            self._setup_layernorms()

    def _setup_layernorms(self):
        """
        Set up non-learnable layernorms for each block.
        Called after _setup_model when num_blocks and embed_dim are available.
        """
        if not hasattr(self, "num_blocks") or not hasattr(self, "embed_dim"):
            return

        if self.layernorm_type == "embed_only":
            # Normalise over embedding dimension only
            self.layernorms = nn.ModuleList(
                [
                    nn.LayerNorm(self.embed_dim, elementwise_affine=False, bias=False)
                    for _ in range(self.num_blocks)
                ]
            )
        elif self.layernorm_type == "embed_and_tokens":
            # No per-layer modules needed for this option
            # Custom normalisation is applied in _apply_layernorm
            self.layernorms = [None] * self.num_blocks

    @abc.abstractmethod
    def _setup_model(self, **kwargs):
        """
        Set up the model with specified configuration.
        This should be implemented by each specific backbone.
        """
        pass

    @abc.abstractmethod
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply model-specific normalisation to input tensor."""
        pass

    def get_model(self):
        """Return the underlying model (from timm or transformers)"""
        return self._model

    def _resize_feature_map(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Resize feature map to target grid size if specified."""
        if self.target_grid_size is None:
            return feature_map

        current_size = feature_map.shape[-2:]
        if current_size == self.target_grid_size:
            return feature_map

        # Reshape to target grid size using interpolation
        return F.interpolate(
            feature_map,
            size=self.target_grid_size,
            mode="bilinear",
            align_corners=False,
        )

    def _unify_tokens(
        self, feature_map: torch.Tensor, cls_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Unify cls tokens and spatial tokens into a single sequence based on cls_token_strategy."""
        batch_size, dim, height, width = feature_map.shape

        # Reshape feature map: [B, D, H, W] -> [B, D, H*W]
        spatial_tokens = feature_map.reshape(batch_size, dim, height * width)

        # Handle cls tokens according to strategy
        if self.cls_token_strategy == "spatial_only":
            # Skip cls tokens entirely
            unified_tokens = spatial_tokens
        elif self.cls_token_strategy == "first_only" and cls_tokens.shape[2] > 1:
            # Keep only the first cls token
            unified_tokens = torch.cat([cls_tokens[:, :, 0:1], spatial_tokens], dim=2)
        elif (
            self.cls_token_strategy == "fixed"
            and cls_tokens.shape[2] != self.num_cls_tokens
        ):
            # Either truncate or pad to get the fixed number
            if cls_tokens.shape[2] > self.num_cls_tokens:
                cls_tokens = cls_tokens[:, :, : self.num_cls_tokens]
            else:
                padding = torch.zeros(
                    (batch_size, dim, self.num_cls_tokens - cls_tokens.shape[2]),
                    device=cls_tokens.device,
                    dtype=cls_tokens.dtype,
                )
                cls_tokens = torch.cat([cls_tokens, padding], dim=2)
            unified_tokens = torch.cat([cls_tokens, spatial_tokens], dim=2)
        else:
            # Default "all" strategy: use all provided cls tokens
            unified_tokens = torch.cat([cls_tokens, spatial_tokens], dim=2)

        # Permute from [B, D, seq_len] to [B, seq_len, D]
        return unified_tokens.permute(0, 2, 1)

    def _separate_tokens(
        self, unified_tokens: torch.Tensor, grid_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate unified tokens back into cls tokens and feature map."""
        batch_size, seq_len, dim = unified_tokens.shape
        h, w = grid_size

        if self.cls_token_strategy == "spatial_only":
            # No cls tokens, all are spatial
            # First permute back to [B, D, seq_len]
            spatial_tokens = unified_tokens.permute(0, 2, 1)
            feature_map = spatial_tokens.reshape(batch_size, dim, h, w)
            cls_tokens = torch.zeros((batch_size, dim, 0), device=unified_tokens.device)
            return feature_map, cls_tokens

        # For all other strategies, determine number of cls tokens
        if self.cls_token_strategy == "fixed":
            num_cls = self.num_cls_tokens
        elif self.cls_token_strategy == "first_only":
            num_cls = 1
        else:  # 'all'
            num_cls = self.get_actual_num_cls_tokens()

        # Extract cls tokens and spatial tokens
        cls_tokens_part = unified_tokens[:, :num_cls, :]  # [B, num_cls, D]
        spatial_tokens = unified_tokens[:, num_cls:, :]  # [B, H*W, D]

        # Permute cls tokens to [B, D, num_cls]
        cls_tokens = cls_tokens_part.permute(0, 2, 1)

        # Reshape spatial tokens to feature map [B, D, H, W]
        # First permute from [B, H*W, D] to [B, D, H*W]
        spatial_tokens = spatial_tokens.permute(0, 2, 1)
        feature_map = spatial_tokens.reshape(batch_size, dim, h, w)

        return feature_map, cls_tokens

    @abc.abstractmethod
    def get_actual_num_cls_tokens(self) -> int:
        """
        Get the actual number of cls tokens in this backbone model.
        To be implemented by specific backbone classes.
        """
        pass

    def _apply_layernorm(
        self, unified_tokens: torch.Tensor, layer_idx: int
    ) -> torch.Tensor:
        """Apply non-learnable layernorm to unified token sequence."""
        if not self.use_layernorm:
            return unified_tokens

        if self.layernorm_type == "embed_only":
            # Normalise over embedding dimension only
            if self.layernorms is not None:
                return self.layernorms[layer_idx](unified_tokens)
            return unified_tokens

        elif self.layernorm_type == "embed_and_tokens":
            # Normalise over both embedding dimension and tokens (dims 1 and 2)
            # Shape: [B, seq_len, D]

            # Calculate mean and variance across both seq_len and embedding dimensions
            # keepdim=True ensures we get shape [B, 1, 1]
            mean = unified_tokens.mean(dim=(1, 2), keepdim=True)
            var = unified_tokens.var(dim=(1, 2), unbiased=False, keepdim=True)

            return (unified_tokens - mean) / torch.sqrt(var + 1e-5)

        return unified_tokens

    def _process_block_output(
        self, output: Tuple[torch.Tensor, torch.Tensor], layer_idx: int = 0
    ) -> torch.Tensor:
        """Process output from a block to standardize the format."""

        feature_map, cls_tokens = output

        # Resize feature map if target size is specified
        if self.target_grid_size is not None:
            feature_map = self._resize_feature_map(feature_map)

        # Unify tokens to standard format
        unified_tokens = self._unify_tokens(feature_map, cls_tokens)

        # Apply layernorm if enabled
        if self.use_layernorm and self.layernorms is not None:
            unified_tokens = self._apply_layernorm(unified_tokens, layer_idx)

        return unified_tokens

    @abc.abstractmethod
    def _get_raw_block_outputs(
        self, x: torch.Tensor
    ) -> List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get raw outputs from each transformer block before any normalisation or resizing.
        This is an internal method that should be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def _get_raw_norm2_outputs(
        self, x: torch.Tensor
    ) -> List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get raw outputs after each norm2 in transformer blocks before any normalisation or resizing.
        This is an internal method that should be implemented by subclasses.
        """
        pass

    def get_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features from all blocks in the model."""
        # Choose between norm2 outputs and block outputs based on flag
        if not self.use_norm2_outputs:
            # Get block outputs (before norm2)
            raw_outputs = self._get_raw_block_outputs(x)
        else:
            # Get norm2 outputs (after norm2)
            raw_outputs = self._get_raw_norm2_outputs(x)

        # Process each layer's output
        processed_outputs = []
        for i, output in enumerate(raw_outputs):
            processed = self._process_block_output(output, layer_idx=i)
            processed_outputs.append(processed)

        return processed_outputs

    @abc.abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the backbone model."""
        pass

    # Note: __call__ is NOT overridden here. nn.Module.__call__ handles
    # hooks, autograd, train/eval mode, etc. Use get_features() directly
    # when feature extraction from all layers is needed.

    def get_embedding_dim(self) -> int:
        """Get the embedding dimension of the backbone."""
        return self.embed_dim

    def get_num_layers(self) -> int:
        """Get the number of transformer layers/blocks in the backbone."""
        return self.num_blocks

    def get_name(self) -> str:
        """Get the descriptive name of the backbone."""
        return self.name

    def _calculate_token_counts(self):
        """
        Calculate and store the number of cls, spatial, and total tokens.
        Called after _setup_model when model attributes are available.
        """
        # Get actual number of cls tokens from the model
        actual_cls_tokens = self.get_actual_num_cls_tokens()

        # Apply cls token strategy
        if self.cls_token_strategy == "spatial_only":
            self.token_counts["cls"] = 0
        elif self.cls_token_strategy == "first_only":
            self.token_counts["cls"] = 1 if actual_cls_tokens > 0 else 0
        elif self.cls_token_strategy == "fixed":
            self.token_counts["cls"] = self.num_cls_tokens
        else:  # "all"
            self.token_counts["cls"] = actual_cls_tokens

        # Calculate spatial tokens based on target_grid_size first (if specified)
        # This takes priority since the feature map will be resized in _process_block_output
        if self.target_grid_size is not None:
            self.token_counts["spatial"] = (
                self.target_grid_size[0] * self.target_grid_size[1]
            )
        # If no target grid size, calculate based on image size and patch size
        elif hasattr(self, "patch_size"):
            h, w = self.image_size
            patch_size = self.patch_size
            if isinstance(patch_size, tuple):
                patch_h, patch_w = patch_size
                self.token_counts["spatial"] = (h // patch_h) * (w // patch_w)
            else:
                self.token_counts["spatial"] = (h // patch_size) * (w // patch_size)
        else:
            raise ValueError(
                "Cannot determine spatial token count without patch size or target grid size."
            )

        # Calculate total tokens
        self.token_counts["total"] = (
            self.token_counts["cls"] + self.token_counts["spatial"]
        )

    def get_num_cls_tokens(self) -> int:
        """
        Get the effective number of cls tokens according to the strategy.
        """
        return self.token_counts["cls"]

    def get_num_spatial_tokens(self) -> int:
        """Get the number of spatial tokens in this backbone."""
        return self.token_counts["spatial"]

    def get_num_total_tokens(self) -> int:
        """Get the total number of tokens (cls + spatial) in this backbone."""
        return self.token_counts["total"]
