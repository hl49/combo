"""
Adapter and backbone loading utilities.

Supports ComBo, linear probe, LoRA, full fine-tuning, and Adapter+ adapters
with ViT backbones.
"""

from omegaconf import OmegaConf

from .adapters.adapter_plus import AdapterPlusAdapter
from .adapters.combo import ComBoAdapter
from .adapters.full_tune import FullTuneAdapter
from .adapters.linear_probe import LinearProbeAdapter
from .adapters.lora import LoRAAdapter
from .backbones.combined_backbones import CombinedBackbone
from .backbones.timm_vit import TimmVisionTransformerBackbone


def load_backbone(backbone_config):
    """
    Load a backbone model based on configuration.

    Supports single backbones and combined (multi-backbone) configurations.
    """
    # Check if we have multiple backbones defined
    if hasattr(backbone_config, "backbones") and backbone_config.backbones:
        backbones_list = backbone_config.backbones

        individual_backbones = []
        backbone_names = []

        for i, sub_config in enumerate(backbones_list):
            name = sub_config.get("name", f"backbone_{i}")
            backbone_names.append(name)
            backbone = _load_individual_backbone(sub_config)
            individual_backbones.append(backbone)

        # If there's only one backbone, return it directly
        if len(individual_backbones) == 1:
            return individual_backbones[0]

        # Create combined backbone with multiple backbones
        return CombinedBackbone(
            backbones=individual_backbones,
            backbone_names=backbone_names,
            name=backbone_config.get("name", None),
            cls_token_strategy=backbone_config.get("cls_token_strategy", "max_pad"),
            use_layernorm=backbone_config.get("use_layernorm", False),
            use_norm2_outputs=backbone_config.get("use_norm2_outputs", False),
        )
    else:
        return _load_individual_backbone(backbone_config)


def _load_individual_backbone(backbone_config):
    """Load a single backbone based on config."""
    backbone_config = OmegaConf.to_container(backbone_config, resolve=True)

    return TimmVisionTransformerBackbone(
        model_path=backbone_config["model_path"],
        target_grid_size=backbone_config.get("target_grid_size", None),
        use_layernorm=backbone_config.get("use_layernorm", False),
        name=backbone_config.get("name", None),
        use_norm2_outputs=backbone_config.get("use_norm2_outputs", False),
        include_register_tokens=backbone_config.get("include_register_tokens", False),
        image_size=backbone_config.get("image_size", (224, 224)),
        cls_token_strategy=backbone_config.get("cls_token_strategy", "all"),
        drop_path_rate=backbone_config.get("drop_path_rate", 0.0),
        init_last_norm_from_pretrained=backbone_config.get(
            "init_last_norm_from_pretrained", False
        ),
    )


def load_adapter(adapter_config, backbone_config, num_classes: int):
    """Load a backbone model with an adapter using config objects."""
    backbone = load_backbone(backbone_config)

    adapter_type = adapter_config.type
    freeze_backbone = adapter_config.get("freeze_backbone", True)
    params = adapter_config.get("params", {})

    # Resolve OmegaConf to plain dict for safe unpacking
    if hasattr(params, "_iter_ex"):  # OmegaConf DictConfig
        params = OmegaConf.to_container(params, resolve=True)

    adapter_map = {
        "combo": ComBoAdapter,
        "linear_probe": LinearProbeAdapter,
        "full_tune": FullTuneAdapter,
        "lora": LoRAAdapter,
        "adapter_plus": AdapterPlusAdapter,
    }

    adapter_cls = adapter_map.get(adapter_type.lower())
    if adapter_cls is None:
        raise ValueError(
            f"Unsupported adapter type: {adapter_type}. "
            f"Supported types: {list(adapter_map.keys())}"
        )

    return adapter_cls(
        backbone=backbone,
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        **params,
    )
