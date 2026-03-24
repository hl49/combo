# Code for ComBo paper

This repository contains the code accompanying our paper:

"Fantastic Features and Where to Find Them: A Probing Method to Combine Features from Multiple Foundation Models"

*Benjamin Ramtoula, Pierre-Yves Lajoie, Paul Newman, Daniele De Martini*

*NeurIPS 2025*

[Project page](https://bramtoula.github.io/combo/), [arXiv](https://arxiv.org/abs/2512.01405), [OpenReview](https://openreview.net/forum?id=PxgIElCohI)

The paper introduces a new probing-based adapter, ComBo, that combines intermediate features from one or more frozen vision transformer (ViT) foundation models. It applies a learned linear projection to compress stacked layer embeddings at each spatial position, then processes the resulting tokens with a lightweight transformer encoder. ComBo does not require backpropagation through backbone models or dataset-specific hyperparameter tuning.

## Overview
The repository provides code to run ComBo to adapt different backbones or combinations of backbones, as well as some baselines.

The code relies on [Hydra](https://hydra.cc/) for configuration management, [PyTorch Lightning](https://www.pytorchlightning.ai/) for training, [Hugging Face Datasets](https://huggingface.co/docs/datasets) for dataset loading, [timm](https://github.com/huggingface/pytorch-image-models) for backbone loading, and [Weights & Biases](https://wandb.ai/) for logging results.

The default configurations (in [configs/](configs/)) provided are meant to reproduce the experimental settings described in the paper for ComBo, but might not be ideal for all baselines.

The codebase also retains some features that were experimental during development but are not used in the approach presented in the paper, such as regularisation schemes or applying dropout to feature maps.

## Installation

```bash
git clone https://github.com/bramtoula/combo.git
cd combo
pip install -r requirements.txt
```

## Usage

### Training with various configurations

The main script for training adapters is [train.py](train.py). Details regarding the implementations are provided below in the **Additional details** section.

By default, it will use the config described in [configs/config.yaml](configs/config.yaml) (ComBo on ViT-B/16 pre-trained on ImageNet-21K on CIFAR-100 from VTAB-1k):
```
python train.py
```

You can use different configurations by specifying the names of the config files (without the .yaml extension) from the different config directories.
For example, you can change the backbone to DINOv2 (ViT-B/14) by specifying the corresponding config file [configs/backbone/dinov2.yaml](configs/backbone/dinov2.yaml):

```
python train.py backbone=dinov2
```

Or change the dataset to Caltech101 from VTAB-1k by specifying the corresponding config file [configs/data/vtab_caltech101.yaml](configs/data/vtab_caltech101.yaml):

```
python train.py data=vtab_caltech101
```

Or train a LoRA adapter instead of ComBo by specifying the corresponding config file [configs/adapter/lora.yaml](configs/adapter/lora.yaml):

```
python train.py adapter=lora
```

Or combine those three by specifying multiple config files:

```
python train.py adapter=lora backbone=dinov2 data=vtab_caltech101
```

Using multiple backbones is managed by creating a config file in [configs/backbone/](configs/backbone/) that specifies the backbones to use. For example, [configs/backbone/multi_backbone.yaml](configs/backbone/multi_backbone.yaml) specifies four backbones used in the paper: SAM, DFN-CLIP, SigLIP, and DINOv2, and the result for CIFAR-100 can be reproduced with:

```
python train.py adapter=combo backbone=multi_backbone data=vtab_cifar100
```

### Overriding hyperparameters

Thanks to Hydra, you can easily override any configuration parameter directly from the command line without modifying the YAML files. For example, you can change the learning rate, the number of training steps, or disable logging to Weights & Biases (which is enabled by default):

```bash
python train.py optimizer.lr=5e-4 training.max_steps=2000 trainer.logger.offline=true
```

### Evaluating on the test set
The default config provided is meant to be used for training on the train800 split of the VTAB-1k datasets and evaluating on the val200 splits. For final testing, a few settings need to be overwritten. We provide a script that can be used to evaluate on the test set:

```bash
bash scripts/evaluate.sh combo in21k vtab_cifar100 [additional_hydra_args]
```

This sets `data.train_split_name="train800val200"` (a pre-defined split in the hosted Hugging Face datasets that combines the 800 training and 200 validation samples) and `trainer.test_after_training=true`. The script also scales training steps from 1300 to 1600 to maintain the same number of epochs (100) with the larger training set.


## Additional details

### Training pipeline
The core training and evaluation stages are managed by PyTorch Lightning. The main logic is implemented in the `ImageClassificationLightningModule` located in [`models/image_classification_module.py`](models/image_classification_module.py). This module handles model initialisation, loss computation (including regularisation), and configuring optimisers and learning rate schedulers.

### General training parameters
General training parameters such as seed, device, optimiser, learning rate, etc. (some of them managed by the PyTorch Lightning trainer) are specified in various config files in [configs/](configs/):
- Training duration: [configs/training/default.yaml](configs/training/default.yaml)
- Optimiser: [configs/optimizer/adamw.yaml](configs/optimizer/adamw.yaml)
- Scheduler: [configs/scheduler/linear_warmup_cosine.yaml](configs/scheduler/linear_warmup_cosine.yaml)
- Trainer (logging, devices, precision, etc.): [configs/trainer/default.yaml](configs/trainer/default.yaml)


### Datasets
The data loaded from the Hugging Face Hub is managed by the VTAB DataModule [`data/vtab_datamodule.py`](data/vtab_datamodule.py) which is a [PyTorch Lightning DataModule](https://lightning.ai/docs/pytorch/stable/data/datamodule.html) that handles loading and preprocessing the data.

Currently, we only provide three of the VTAB-1k datasets publicly uploaded to the Hugging Face Hub (`vtab_cifar100`, `vtab_caltech101`, `vtab_patch_camelyon`) due to licensing restrictions. For any other datasets (including the remaining VTAB tasks), you must manually supply or upload the data yourself.

For detailed instructions on the expected data format, how to prepare the remaining unhosted VTAB datasets (e.g., using `scripts/upload_vtab_to_hf.py`), or how to integrate entirely custom datasets, please refer to [data/DATA_FORMAT.md](data/DATA_FORMAT.md).

### Backbones

The base backbone class specifying common functionalities and expected interface is [models/backbones/backbones_base.py](models/backbones/backbones_base.py).

Backbones available in the [timm](https://github.com/huggingface/pytorch-image-models) library are managed via [models/backbones/timm_vit.py](models/backbones/timm_vit.py).

The functionality and interfaces to manage multiple backbones are provided by [models/backbones/combined_backbones.py](models/backbones/combined_backbones.py).

The use of a backbone or multiple backbones can be specified in the config files, which are parsed and used to instantiate the backbone(s) in `load_backbone` in [models/adapter_loader.py](models/adapter_loader.py).

Here is a summary of configurations provided in this repository:

| Config name | Models | `timm` identifier(s) | Config file |
|---|---|---|---|
| `in21k` | ViT-B/16 (ImageNet-21K) | `timm/vit_base_patch16_224.orig_in21k` | `configs/backbone/in21k.yaml` |
| `sam` | SAM ViT-B/16 | `timm/samvit_base_patch16.sa1b` | `configs/backbone/sam.yaml` |
| `dfn_clip` | DFN-CLIP ViT-B/16 | `timm/vit_base_patch16_clip_224.dfn2b` | `configs/backbone/dfn_clip.yaml` |
| `siglip` | SigLIP ViT-B/16 | `timm/vit_base_patch16_siglip_224.webli` | `configs/backbone/siglip.yaml` |
| `dinov2` | DINOv2 ViT-B/14 (w/ registers) | `timm/vit_base_patch14_reg4_dinov2.lvd142m` | `configs/backbone/dinov2.yaml` |
| `multi_backbone` | SAM + DFN-CLIP + SigLIP + DINOv2 | *(See individual configs)* | `configs/backbone/multi_backbone.yaml` |

Adding new ViT backbones available in `timm` should be possible by creating new configs in `configs/backbone/` that specify the `timm` identifier and the desired configuration.


### Adapters

Adapters that work on top of backbones are implemented in [models/adapters/](models/adapters/), using the common interface provided by [models/adapters/adapter_base.py](models/adapters/adapter_base.py).
They are meant to be provided with a backbone or multiple backbones (if compatible) and update them with modules to learn or extract features to process to produce a downstream task prediction.

Here is a summary of the adapter configurations provided:

| Config name | Adapter | Config file |
|---|---|---|
| `combo` | ComBo (Ours) | `configs/adapter/combo.yaml` |
| `adapter_plus` | Adapter+ | `configs/adapter/adapter_plus.yaml` |
| `full_tune` | Full Fine-Tuning | `configs/adapter/full_tune.yaml` |
| `linear_probe` | Linear Probing | `configs/adapter/linear_probe.yaml` |
| `lora` | LoRA | `configs/adapter/lora.yaml` |

*Note: Not all adapters are compatible with multiple backbones.*

In particular, our proposed ComBo adapter is implemented in [models/adapters/combo.py](models/adapters/combo.py).

Please note that if you plan to reimplement ComBo in your own codebase, the steps for extracting features, processing feature maps, and applying the adapter are spread across multiple files, including [models/backbones/timm_vit.py](models/backbones/timm_vit.py), [models/backbones/backbones_base.py](models/backbones/backbones_base.py), [models/backbones/combined_backbones.py](models/backbones/combined_backbones.py) (if using multiple backbones), and [models/adapters/combo.py](models/adapters/combo.py).

### Logging

Training is logged via [Weights & Biases](https://wandb.ai/), configured in `configs/trainer/default.yaml`. WandB logging is enabled by default (`offline: false`). To disable it, set `trainer.logger.offline=true` from the command line, or modify the config file.


# Reference

If you make use of our project, please consider citing:

```bibtex
@inproceedings{ramtoula2025combo,
    title={Fantastic Features and Where to Find Them: A Probing Method to combine Features from Multiple Foundation Models},
    author={Benjamin Ramtoula and Pierre-Yves Lajoie and Paul Newman and Daniele De Martini},
    booktitle={NeurIPS},
    year={2025}
}
```

# Acknowledgements

- [VTAB benchmark](https://github.com/google-research/task_adaptation)
- [timm library](https://github.com/huggingface/pytorch-image-models)
- [PyTorch Lightning](https://lightning.ai/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets)
- [Weights & Biases](https://wandb.ai/)
- [Adapters Strike Back codebase](https://github.com/visinf/adapter_plus)
