"""
Train an image classifier using a ComBo adapter (or baseline adapter) on a
classification dataset.

Uses Hydra for configuration management and PyTorch Lightning for training.

Examples:
    # Train ComBo on CIFAR-100 with a single ViT-B/16 backbone
    python train.py adapter=combo backbone=in21k data=vtab_cifar100

    # Train ComBo with multiple backbones
    python train.py adapter=combo backbone=multi_backbone data=vtab_eurosat

    # Train a linear probe baseline
    python train.py adapter=linear_probe backbone=in21k data=vtab_cifar100

    # Train with custom hyperparameters
    python train.py adapter=combo backbone=in21k data=vtab_cifar100 \\
        training.max_steps=2000 optimizer.lr=5e-4
"""

import logging
import warnings

import hydra
import lightning as L
import lightning.pytorch.callbacks as pl_callbacks
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from data.vtab_datamodule import VTABDataModule
from models.image_classification_module import ImageClassificationLightningModule

# Suppress timm internal deprecation warning (transitive import)
warnings.filterwarnings(
    "ignore", message="Importing from timm.models.helpers is deprecated"
)

# Suppress verbose httpx logs from huggingface datasets
logging.getLogger("httpx").setLevel(logging.WARNING)


def get_model(cfg):
    """Create a model from config."""
    return ImageClassificationLightningModule(cfg)


def get_data(data_cfg):
    """Create a data module from config."""
    return VTABDataModule(
        dataset_name=data_cfg.dataset_name,
        batch_size=data_cfg.batch_size,
        eval_batch_size=data_cfg.eval_batch_size,
        shuffle_val_loader=data_cfg.shuffle_val_loader,
        num_workers=data_cfg.num_workers,
        train_split_name=data_cfg.train_split_name,
        val_split_name=data_cfg.val_split_name,
        test_split_name=data_cfg.test_split_name,
        force_resize=data_cfg.force_resize,
        interpolation=data_cfg.interpolation,
        limit_train_samples=data_cfg.limit_train_samples,
    )


def get_logger(cfg):
    """Create WandB logger from config."""
    return WandbLogger(
        name=cfg.trainer.logger.name,
        project=cfg.trainer.logger.project,
        log_model=cfg.trainer.logger.log_model,
        save_dir=cfg.trainer.logger.save_dir,
        offline=cfg.trainer.logger.offline,
    )


def get_callbacks(cfg):
    """Create callbacks from config."""
    callbacks = []
    for callback in cfg.trainer.callbacks:
        callback_class = getattr(pl_callbacks, callback.name)
        callbacks.append(callback_class(**callback.init_args))
    return callbacks


def train(cfg):
    """
    Train a model with the given configuration.
    """
    model = get_model(cfg)
    data = get_data(cfg.data)

    L.seed_everything(cfg.trainer.seed_everything)

    logger = get_logger(cfg)

    logger.experiment.config.update(
    OmegaConf.to_container(cfg, resolve=True),
    allow_val_change=True,
        )

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        precision=cfg.trainer.precision,
        # logger=get_logger(cfg),
        logger=logger,
        callbacks=get_callbacks(cfg),
        max_epochs=cfg.training.max_epochs,
        min_epochs=cfg.training.min_epochs,
        max_steps=cfg.training.max_steps,
        min_steps=cfg.training.min_steps,
        max_time=cfg.training.max_time,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        overfit_batches=cfg.trainer.overfit_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        deterministic=cfg.trainer.deterministic,
        benchmark=cfg.trainer.benchmark,
        inference_mode=cfg.trainer.inference_mode,
        profiler=cfg.trainer.profiler,
        detect_anomaly=cfg.trainer.detect_anomaly,
        barebones=cfg.trainer.barebones,
        plugins=cfg.trainer.plugins,
        sync_batchnorm=cfg.trainer.sync_batchnorm,
        reload_dataloaders_every_n_epochs=cfg.trainer.reload_dataloaders_every_n_epochs,
        default_root_dir=cfg.trainer.default_root_dir,
    )

    torch.set_float32_matmul_precision("medium")

    # Train the model
    trainer.fit(model, data)

    # Validate
    val_metrics = trainer.validate(model, data)

    # Test
    test_metrics = None
    if cfg.trainer.test_after_training:
        test_metrics = trainer.test(model, data)

    return model, data, test_metrics


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
