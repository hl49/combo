# Expected Data Format

This document details the data format expectations for the ComBo repository. If you are running tests on VTAB tasks that are not pre-hosted (any task other than `vtab_cifar100`, `vtab_caltech101`, and `vtab_patch_camelyon`), you must obtain the original data yourself, format it as described below, and update the corresponding configuration template's `dataset_name` field.

## HuggingFace Dataset Format

The provided code relies on the [Hugging Face Datasets](https://huggingface.co/docs/datasets) library for data ingestion. To use it as is, you need to provide your data in a compatible format with the following structure:

### Required Columns

| Column  | Type      | Description                          |
|---------|-----------|--------------------------------------|
| `image` | PIL Image | The input image                      |
| `label` | int       | Integer class label (0 to num_classes-1) |

### Required Splits

For VTAB-1k experiments, the following splits are expected:

| Split            | Description                           | Typical Size |
|------------------|---------------------------------------|--------------|
| `train800`       | Training set                          | 800 images   |
| `val200`         | Validation set                        | 200 images   |
| `train800val200` | Combined training and validation sets | 1000 images  |
| `test`           | Test set for final evaluation         | varies       |

*Note: These expected split names can be customised via the data configuration file.* 

## Supporting unhosted VTAB datasets

To help with preparing the unhosted VTAB tasks, we provide a script for inspiration (`scripts/upload_vtab_to_hf.py`). This script relies on the original VTAB benchmark codebase (see [the original repository](https://github.com/google-research/task_adaptation)) and assumes the datasets have been prepared locally. Please note that the script is not guaranteed to work out of the box and may require some modifications to fit your specific setup.

## Creating a Custom Hugging Face Dataset

To create a new Hugging Face Dataset form scratch, please refer to the official documentation [here](https://huggingface.co/docs/datasets/index).

Once you have created your dataset, you can use it by creating a config file in `configs/data/` for your dataset:

```yaml
num_classes: 10  # Number of classes in your dataset
dataset_name: your-username/your-dataset-name  # Hugging Face dataset name or path
batch_size: 64
eval_batch_size: 32
shuffle_val_loader: false
num_workers: 4
train_split_name: train      # Name of the training split
val_split_name: validation   # Name of the validation split
test_split_name: test        # Name of the test split
force_resize: [224, 224]
interpolation: bilinear
limit_train_samples: null     # Set to an integer to limit training samples
```

Then run training with:
```bash
python train.py data=your_config_name
```

## Alternative: Using a Custom DataModule

To support your own local data without using Hugging Face Datasets at all, you can alternatively create a custom DataModule:
1. Inherit from [`pl.LightningDataModule`](https://lightning.ai/docs/pytorch/stable/data/datamodule.html).
2. Implement the same interface methods as `vtab_datamodule.py`.
3. Update `train.py` to use your custom DataModule instead of the provided one.
