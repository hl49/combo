import abc
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ..backbones.backbones_base import BaseVisionTransformerBackbone
from ..backbones.combined_backbones import CombinedBackbone


class BaseAdapter(nn.Module, abc.ABC):
    """
    Abstract base class for adapters that wrap backbone models.

    Adapters modify or augment the behaviour of backbone models for specific tasks
    through methods like probing or fine-tuning.
    """

    def __init__(
        self,
        backbone: Union[BaseVisionTransformerBackbone, CombinedBackbone],
        num_classes: int,
        freeze_backbone: bool = True,
        **kwargs,
    ):
        """Initialise the adapter."""
        super().__init__()

        self.backbone = backbone
        self.num_classes = num_classes

        # Get key backbone properties
        self.backbone_embedding_dim = self.backbone.get_embedding_dim()
        self.backbone_num_cls_tokens = self.backbone.get_num_cls_tokens()
        self.backbone_num_total_tokens = self.backbone.get_num_total_tokens()
        self.backbone_num_layers = self.backbone.get_num_layers()

        # Apply backbone freezing if requested
        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()

        # Initialise step counter for tracking training progress
        self.current_step = 0
        self.current_epoch = 0
        self.total_steps = kwargs.get("total_steps", 0)
        self.total_epochs = kwargs.get("total_epochs", 0)

        # Set up adapter-specific components
        self._setup_adapter(**kwargs)

    @abc.abstractmethod
    def _setup_adapter(self, **kwargs):
        """
        Set up adapter-specific components.
        This should be implemented by each specific adapter type.
        """
        pass

    def update_step_counter(
        self,
        step: int,
        epoch: int = None,
        total_steps: int = None,
        total_epochs: int = None,
    ):
        """Update the step counter for the adapter."""
        self.current_step = step

        if epoch is not None:
            self.current_epoch = epoch

        if total_steps is not None:
            self.total_steps = total_steps

        if total_epochs is not None:
            self.total_epochs = total_epochs

        # Call hook for subclasses to implement custom behaviour
        self.on_step_update(step, epoch)

    def on_step_update(self, step: int, epoch: int = None):
        """Hook called when the step counter is updated.
        Subclasses can override this to implement step-dependent behaviour.
        """
        pass

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the adapter."""
        pass

    @abc.abstractmethod
    def get_adapter_parameters(self) -> Dict[str, Any]:
        """Get parameters specific to this adapter for serialization."""
        pass

    def load_adapter_parameters(self, params: Dict[str, Any]):
        """Load adapter parameters from a dictionary."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_regularization_losses(
        self,
        l1_scale: float = 0.0,
        l2_scale: float = 0.0,
        gl_scale: float = 0.0,
        struct_scale: float = 0.0,
        model_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate regularization losses for the adapter."""
        # Default implementation returns zero losses
        device = next(self.parameters()).device
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero, zero, zero

    def get_data_to_log(self) -> Dict[str, Any]:
        """Get data that should be logged during training."""
        # Default implementation returns an empty dictionary
        return {}

    def save_checkpoint(self, path: str):
        """Save the adapter and its backbone as a complete model that can be loaded later."""
        # Get class names for recreation
        adapter_class_name = self.__class__.__name__
        backbone_class_name = self.backbone.__class__.__name__

        # Save the model state
        torch.save(
            {
                "adapter_type": adapter_class_name,
                "adapter_params": self.get_adapter_parameters(),
                "backbone_type": backbone_class_name,
                "backbone_name": self.backbone.name
                if hasattr(self.backbone, "name")
                else None,
                "backbone_model_path": self.backbone.model_path,
                "backbone_params": {
                    "target_grid_size": getattr(
                        self.backbone, "target_grid_size", None
                    ),
                    "use_layernorm": getattr(self.backbone, "layernorm_type", False),
                    "use_norm2_outputs": getattr(
                        self.backbone, "use_norm2_outputs", False
                    ),
                    "cls_token_strategy": getattr(
                        self.backbone, "cls_token_strategy", "all"
                    ),
                },
                "model_state_dict": self.state_dict(),
                "num_classes": self.num_classes,
                "embedding_dim": self.backbone_embedding_dim,
                "num_layers": self.backbone_num_layers,
            },
            path,
        )
        print(f"Model saved to {path}")

    @classmethod
    def load_checkpoint(cls, path: str, backbone=None, **kwargs):
        """Load an adapter from a checkpoint."""
        from ..adapter_loader import load_adapter, load_backbone

        # Load the checkpoint
        checkpoint = torch.load(path, weights_only=False)

        # If no backbone provided, load it from the checkpoint
        if backbone is None:
            backbone_config = OmegaConf.create(
                {
                    "model_path": checkpoint["backbone_model_path"],
                    "name": checkpoint["backbone_name"],
                    **checkpoint["backbone_params"],
                }
            )
            backbone = load_backbone(backbone_config)

        # Create the adapter via config-based loading
        adapter_type = checkpoint["adapter_type"]
        adapter_params = {**checkpoint["adapter_params"], **kwargs}
        num_classes = checkpoint["num_classes"]

        # Map class names to config type strings
        class_to_type = {
            "ComBoAdapter": "combo",
            "LinearProbeAdapter": "linear_probe",
            "LoRAAdapter": "lora",
            "FullTuneAdapter": "full_tune",
            "AdapterPlusAdapter": "adapter_plus",
        }
        config_type = class_to_type.get(adapter_type, adapter_type.lower())

        adapter_config = OmegaConf.create(
            {"type": config_type, "freeze_backbone": True, "params": adapter_params}
        )
        backbone_config = OmegaConf.create(
            {
                "model_path": checkpoint["backbone_model_path"],
                "name": checkpoint["backbone_name"],
                **checkpoint["backbone_params"],
            }
        )

        adapter = load_adapter(adapter_config, backbone_config, num_classes)

        # Load the state dict
        adapter.load_state_dict(checkpoint["model_state_dict"])

        return adapter
