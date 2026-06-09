"""
Image Classification Lightning Module.

Handles training, validation, and testing of adapter-based image classifiers
using PyTorch Lightning.
"""

import lightning.pytorch as pl
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import Accuracy

from .adapter_loader import load_adapter
from pathlib import Path


def get_model(config):
    """Create model from config objects."""
    return load_adapter(
        adapter_config=config.adapter,
        backbone_config=config.backbone,
        num_classes=config.data.num_classes,
    )


class ImageClassificationLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # Pass OmegaConf config directly — it supports attribute access
        # (e.g. self.hparams.optimizer.weight_decay)
        self.save_hyperparameters(config, logger=False)
        self.model = get_model(config)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(
            task="multiclass", num_classes=config.data.num_classes
        )
        self.val_acc = Accuracy(task="multiclass", num_classes=config.data.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=config.data.num_classes)
        self.best_val_acc = 0.0
        self.best_val_acc_epoch_idx = 0
        self.best_val_acc_step_idx = 0

        # Regularization config
        reg = config.get("regularization", {})
        self.l1_loss_scale = reg.get("l1_loss_scale", 0.0)
        self.l2_loss_scale = reg.get("l2_loss_scale", 0.0)
        self.gl_loss_scale = reg.get("gl_loss_scale", 0.0)
        self.struct_loss_scale = reg.get("struct_loss_scale", 0.0)
        self.model_loss_scale = reg.get("model_loss_scale", 0.0)
        
        # Pruning / saliency config
        # pruning = config.model.pruning
        pruning = config.get("pruning", {})
        # self.output_path = Path(pruning.get("output_path", "outputs"))
        output_dir = Path(pruning.get("output_path", "outputs"))
        dataset_name = config.data.dataset_name.split("/")[-1]
        seed = config.trainer.seed_everything
        self.output_path = output_dir / f'{dataset_name}_seed_{seed}.csv'
        self.warmup_iters = int(pruning.get("warmup_iters", 0.2) * config.training.max_steps)
        self.min_alive = pruning.get("min_alive", 4)
        self.ema_momentum = pruning.get("ema_momentum", 0.9)
        self.log_interval = pruning.get("log_interval", 5)

        # Access model's modules
        self._conv = self.model.feat_transformer.patch_embed.proj
        ch_in = self.model.feat_transformer.in_chans
        self.saliency_ema = torch.zeros(ch_in)
        self._captured_conv_grad = None
        def _capture_conv_grad(grad):
            self._captured_conv_grad = grad.detach().clone()
        self._conv.weight.register_hook(_capture_conv_grad)
        
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            f.write('iter,source,ch_idx,saliency,ema_saliency,grad_mag,weight_mag\n')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]

        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        self.train_acc(outputs, y)

        # Compute regularization if needed
        if (
            self.l1_loss_scale > 0.0
            or self.l2_loss_scale > 0.0
            or self.gl_loss_scale > 0.0
            or self.struct_loss_scale > 0.0
            or self.model_loss_scale > 0.0
        ):
            l1_reg_loss, l2_reg_loss, gl_reg_loss, struct_loss, model_loss = (
                self.model.get_regularization_losses(
                    self.l1_loss_scale,
                    self.l2_loss_scale,
                    self.gl_loss_scale,
                    self.struct_loss_scale,
                    self.model_loss_scale,
                )
            )
            reg_loss = (
                l1_reg_loss + l2_reg_loss + gl_reg_loss + struct_loss + model_loss
            )
        else:
            reg_loss = 0.0

        total_loss = loss + reg_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )

        # Update step counter in the adapter if available
        if hasattr(self.model, "update_step_counter"):
            self.model.update_step_counter(
                step=self.global_step,
                epoch=self.current_epoch,
                total_steps=self.hparams.training.max_steps,
                total_epochs=self.hparams.get("max_epochs", 0),
            )

        return total_loss
    
    def on_after_backward(self) -> None:
        if self._captured_conv_grad is not None:
            # Update EMA of saliency scores
            with torch.no_grad():
                w = self._conv.weight.detach().cpu()
                gw = self._captured_conv_grad.cpu()
                saliency = (gw * w).pow(2).sum(dim=[0, 2, 3])   # [48]
                self.saliency_ema = (
                    self.ema_momentum * self.saliency_ema
                    + (1 - self.ema_momentum) * saliency
                )
        
        if self.global_step % self.log_interval == 0 and self._captured_conv_grad is not None:
            with open(self.output_path, 'a') as f:
                with torch.no_grad():
                    w  = self._conv.weight.detach().cpu()
                    gw = self._captured_conv_grad.cpu()
                    saliency_c = (gw * w).pow(2).sum(dim=[0, 2, 3])
                    grad_mag   = gw.pow(2).sum(dim=[0, 2, 3])
                    weight_mag = w.pow(2).sum(dim=[0, 2, 3])
                    for ch in range(len(saliency_c)):
                        f.write(
                            f'{self.global_step},conv,{ch},'
                            f'{saliency_c[ch].item():.6e},'
                            f'{self.saliency_ema[ch].item():.6e},'
                            f'{grad_mag[ch].item():.6e},'
                            f'{weight_mag[ch].item():.6e}\n'
                        )


    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]

        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        self.val_acc.update(outputs, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_val_acc_epoch_idx = self.current_epoch
            self.best_val_acc_step_idx = self.global_step
        self.log("valid_acc_epoch", val_acc)
        self.log("best_valid_acc_epoch", self.best_val_acc)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]

        outputs = self.model(x)
        self.test_acc.update(outputs, y)

    def on_test_epoch_end(self):
        test_acc = self.test_acc.compute()
        self.log("test_acc", test_acc)
        self.test_acc.reset()

    def configure_optimizers(self):
        # Selective weight decay
        use_selective_weight_decay = self.hparams.optimizer.get(
            "use_selective_weight_decay", False
        )

        if use_selective_weight_decay:
            decay_params = []
            no_decay_params = []
            exclude_from_weight_decay = set()

            def _collect_no_decay_params(module, parent_name=""):
                module_no_decay = set()
                if hasattr(module, "no_weight_decay") and callable(
                    getattr(module, "no_weight_decay")
                ):
                    for name in module.no_weight_decay():
                        if parent_name:
                            module_no_decay.add(f"{parent_name}.{name}")
                        else:
                            module_no_decay.add(name)
                exclude_from_weight_decay.update(module_no_decay)
                for child_name, child_module in module.named_children():
                    child_path = (
                        f"{parent_name}.{child_name}" if parent_name else child_name
                    )
                    _collect_no_decay_params(child_module, child_path)

            _collect_no_decay_params(self.model)

            for name, param in self.model.named_parameters():
                if (
                    name in exclude_from_weight_decay
                    or name.endswith(".bias")
                    or name.endswith(".scaling")
                    or param.ndim <= 1
                ):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            param_groups = [
                {
                    "params": decay_params,
                    "weight_decay": self.hparams.optimizer.weight_decay,
                },
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
        else:
            param_groups = self.model.parameters()

        # Create optimizer
        opt_name = self.hparams.optimizer.name
        if opt_name == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=self.hparams.optimizer.lr,
                weight_decay=self.hparams.optimizer.weight_decay
                if not use_selective_weight_decay
                else 0.0,
            )
        elif opt_name == "sgd":
            optimizer = torch.optim.SGD(
                param_groups,
                lr=self.hparams.optimizer.lr,
                momentum=self.hparams.optimizer.get("momentum", 0.9),
                weight_decay=self.hparams.optimizer.weight_decay
                if not use_selective_weight_decay
                else 0.0,
            )
        elif opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.hparams.optimizer.lr,
                weight_decay=self.hparams.optimizer.weight_decay
                if not use_selective_weight_decay
                else 0.0,
                betas=(
                    self.hparams.optimizer.get("beta1", 0.9),
                    self.hparams.optimizer.get("beta2", 0.999),
                ),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")

        out = {"optimizer": optimizer}

        if self.hparams.scheduler.name != "none":
            warmup_start_lr_factor = self.hparams.scheduler.get(
                "warmup_start_lr_factor", 0.0
            )
            eta_min_factor = self.hparams.scheduler.get("eta_min_factor", 0.0)
            warmup_start_lr = (
                self.hparams.optimizer.lr / warmup_start_lr_factor
                if warmup_start_lr_factor > 0.0
                else 0.0
            )
            eta_min = (
                self.hparams.optimizer.lr / eta_min_factor
                if eta_min_factor > 0.0
                else 0.0
            )

            if self.hparams.scheduler.name == "cosine" or (
                self.hparams.scheduler.name == "linear_warmup_cosine"
                and self.hparams.scheduler.warmup_steps == 0
            ):
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.hparams.training.max_steps,
                    eta_min=eta_min,
                )
            elif self.hparams.scheduler.name == "linear_warmup_cosine":
                if isinstance(self.hparams.scheduler.warmup_steps, float):
                    warmup_steps = int(
                        self.hparams.scheduler.warmup_steps
                        * self.hparams.training.max_steps
                    )
                else:
                    warmup_steps = self.hparams.scheduler.warmup_steps
                scheduler = LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=warmup_steps,
                    max_epochs=self.hparams.training.max_steps,
                    warmup_start_lr=warmup_start_lr,
                    eta_min=eta_min,
                )

            out["lr_scheduler"] = {
                "scheduler": scheduler,
                "interval": "step",
            }

        return out
