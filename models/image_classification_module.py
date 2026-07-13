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
from typing import Any


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
        max_group = pruning.get("max_group", 0)
        self.ema_momentum = pruning.get("ema_momentum", 0.9)
        self.log_interval = int(pruning.get("log_interval", 5))
        self.criteria = pruning.get("criteria", None)
        criteria_list = ['group', 'ratio', 'greedy', 'model']
        assert self.criteria in criteria_list, f"criteria must be one of {criteria_list}"
        self.backbone_slices = {
            "backbone_0": slice(0, 12),
            "backbone_1": slice(12, 24),
            "backbone_2": slice(24, 36),
            "backbone_3": slice(36, 48),
        }
        self.slots = {f"backbone_{i}": self.min_alive for i in range(len(self.backbone_slices.keys()))}
        self._pruned = False
        self.topk_idx=[]

        # Access model's modules
        self._proj = self.model.feat_transformer.patch_embed.proj
        # self._conv = self.model.feat_transformer.patch_embed.proj
        self._conv = getattr(self._proj, "conv", self._proj)
        ch_in = self.model.feat_transformer.in_chans
        self.saliency_ema = torch.zeros(ch_in, dtype=torch.float64)
        self._captured_conv_grad = None
        # def _capture_conv_grad(grad):
        #     self._captured_conv_grad = grad.detach().clone()
        # self._conv.weight.register_hook(_capture_conv_grad)
        # for name, module in self.model.feat_transformer.named_modules():
        #     if name.startswith("conv"):
        #         print(name, module)
        #         break

        if self.model.sigmoid_gates:
            def _capture_gate_grad(grad):
                self._captured_gate_grad = grad.detach().clone()
            self._proj.gate_logits.register_hook(_capture_gate_grad)
            self.gates_saliency_ema = torch.zeros(ch_in, dtype=torch.float64)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            f.write('iter,source,ch_idx,saliency,ema_saliency,grad_mag,weight_mag\n')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.global_step == self.warmup_iters: 
            self._apply_pruning()
            # 
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

    def on_before_optimizer_step(self, optimizer) -> None:
        # Update EMA of saliency scores
        with torch.no_grad():
            w = self._conv.weight.detach().clone().cpu().double()
            # gw = self._captured_conv_grad.cpu()
            gw = self._conv.weight.grad.detach().clone().cpu().double()
            gw = torch.nan_to_num(gw, nan=0.0, posinf=0.0, neginf=0.0)
            saliency = (gw * w).pow(2).sum(dim=[0, 2, 3])   # [48]
            self.saliency_ema = (
                self.ema_momentum * self.saliency_ema
                + (1 - self.ema_momentum) * saliency
            )
            # finite_mask = torch.isfinite(saliency)
            # self.saliency_ema[finite_mask] = (
            #     self.ema_momentum * self.saliency_ema[finite_mask]
            #     + (1 - self.ema_momentum) * saliency[finite_mask]
            # )
            if self.model.sigmoid_gates:
                gl = self._proj.gate_logits.detach().clone().cpu().double()
                ggl = self._proj.gate_logits.grad.detach().clone().cpu().double()
                ggl = torch.nan_to_num(ggl, nan=0.0, posinf=0.0, neginf=0.0)
                gates_saliency = (ggl * gl).pow(2)   # [48]
                self.gates_saliency_ema = (
                    self.ema_momentum * self.gates_saliency_ema
                    + (1 - self.ema_momentum) * gates_saliency
                )
        
        if self.global_step % self.log_interval == 0:
            # ema_log = {
            #     f"saliency_ema/ch_{ch:02d}": self.saliency_ema[ch].item()
            #     for ch in range(len(self.saliency_ema))
            # }
            self.logger.experiment.log({
                **{
                    f"saliency_ema/ch_{ch:02d}": self.saliency_ema[ch].item()
                    for ch in range(len(self.saliency_ema))
                },
                "trainer/global_step": self.global_step,
            })

            # self.log_dict(ema_log, on_step=True)
            with open(self.output_path, 'a') as f:
                with torch.no_grad():
                    grad_mag   = gw.pow(2).sum(dim=[0, 2, 3])
                    weight_mag = w.pow(2).sum(dim=[0, 2, 3])
                    for ch in range(len(saliency)):
                        f.write(
                            f'{self.global_step},conv,{ch},'
                            f'{saliency[ch].item():.6e},'
                            f'{self.saliency_ema[ch].item():.6e},'
                            f'{grad_mag[ch].item():.6e},'
                            f'{weight_mag[ch].item():.6e}\n'
                        )
                    if self.model.sigmoid_gates:
                        ggl_mag = ggl.pow(2)
                        soft = torch.sigmoid(gl)
                        for ch in range(len(self.gates_saliency_ema)):
                            f.write(
                                f'{self.global_step},gate,{ch},'
                                f'{gates_saliency[ch].item():.6e},'
                                f'{self.gates_saliency_ema[ch].item():.6e},'
                                f'{ggl_mag[ch].item():.6e},'
                                f'{soft[ch].item():.6e}\n'
                            )

    def _compute_ratio_slots(self):
        """
        Allocate min_alive slots across backbones proportional to their
        cumulative saliency. Uses largest-remainder to guarantee exact sum.
        Backbones with low saliency may receive 0 slots and be fully pruned.
        """
        total_saliency = self.saliency_ema.sum().item()
        slots          = {}
        remainders     = {}
        ratio_list     = []

        for name, slc in self.backbone_slices.items():
            group_sal        = self.saliency_ema[slc].sum().item()
            ratio            = group_sal / (total_saliency + 1e-40)
            exact            = ratio * self.min_alive
            slots[name]      = int(exact)
            remainders[name] = exact - slots[name]
            ratio_list.append(round(ratio, 3))

        # runner.logger.info(
        #     f'[{self._tag()}] exact slots: {exact_list}'
        # )

        # # Distribute remaining slots to groups with largest remainders
        allocated = sum(slots.values())
        remaining = self.min_alive - allocated
        for name in sorted(remainders, key=remainders.get, reverse=True):
            if remaining <= 0:
                break
            slots[name] += 1
            remaining   -= 1

        return slots, ratio_list


    # def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
    def _apply_pruning(self) -> None:
        self.topk_idx = []

        if self.criteria == 'ratio':
            self.slots, ratio_list = self._compute_ratio_slots()
            msg = (f"[SaliencyPruning] step={self.global_step}, "
            f"slots={self.slots}, ratios={ratio_list}, total={sum(self.slots.values())}")
            self.logger.experiment.log({
                "pruning/summary": msg,
                "pruning/total_alive": sum(self.slots.values()),
                "trainer/pruning_step": self.global_step,
            })
            print(f"[INFO]: {msg}")

        if self.criteria in ('group', 'ratio'):
            # Both use slot-constrained greedy selection
            remaining_slots = dict(self.slots)   # copy — will be decremented
            top_idx         = torch.argsort(self.saliency_ema, descending=True)

            for ch in top_idx:
                backbone_idx  = int(ch.item() // 12)
                backbone_name = f"backbone_{backbone_idx}"
                if remaining_slots.get(backbone_name, 0) > 0:
                    self.topk_idx.append(ch.item())
                    remaining_slots[backbone_name] -= 1
                if self.criteria == 'group' and (len(self.topk_idx) >= self.min_alive*len(self.backbone_slices)):
                    break
                
                if self.criteria == 'ratio' and (len(self.topk_idx) >= self.min_alive):
                    break
                
                # if len(self.topk_idx) >= self.min_alive*len(self.backbone_slices):

        if self.criteria == 'model':
            # ratios = {}
            # # Keep top-k channels per backbone group independently
            # for name, slc in self.backbone_slices.items():
            #     group_saliency = self.saliency_ema[slc].sum().item()
            #     ratio          = group_saliency / (self.saliency_ema.sum().item() + 1e-40)
            #     ratios[name] = ratio
            _, ratio_list = self._compute_ratio_slots()
            ratio_list = torch.tensor(ratio_list)
            backbone_rank = torch.argsort(ratio_list, descending=True)
            # Sort backbones by ratio
            for idx in backbone_rank[:self.min_alive]:
                slc=self.backbone_slices[f"backbone_{idx}"]
                self.topk_idx.extend(range(slc.start, slc.stop))
        
        elif self.criteria == 'greedy':
            top_idx = torch.argsort(self.saliency_ema, descending=True)
            self.topk_idx = top_idx[:self.min_alive].tolist()

        prune_mask = torch.ones(len(self.saliency_ema), dtype=torch.bool)
        prune_mask[self.topk_idx] = False
        self._prune_mask = prune_mask.to(self._conv.weight.device)
        self.saliency_ema[self._prune_mask.cpu()] = 0.0  # zero out pruned channels in EMA for logging

        # Set gates to 0
        with torch.no_grad():
            self._conv.weight[:, self._prune_mask,:,:] = 0.0
            if self.model.sigmoid_gates:
                self._proj.gate_logits[self._prune_mask] = -1e6   # large negative to zero out after sigmoid
                self.gates_saliency_ema[self._prune_mask.cpu()] = 0.0

        optimizer = self.optimizers()
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p is self._conv.weight and p in optimizer.state:
                    state = optimizer.state[p]
                    if "exp_avg" in state:
                        state["exp_avg"][:, self._prune_mask, :, :] = 0.0
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"][:, self._prune_mask, :, :] = 0.0
                if self.model.sigmoid_gates and p is self._proj.gate_logits and p in optimizer.state:
                    state = optimizer.state[p]
                    if "exp_avg" in state:
                        state["exp_avg"][self._prune_mask] = 0.0
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"][self._prune_mask] = 0.0

        def _zero_pruned_grads(grad):
            grad[:, self._prune_mask, :, :] = 0.0
            return grad
        self._conv.weight.register_hook(_zero_pruned_grads)
        if self.model.sigmoid_gates:
            def _zero_pruned_gate_grads(grad):
                grad[self._prune_mask] = 0.0
                return grad
            self._proj.gate_logits.register_hook(_zero_pruned_gate_grads)

        self.logger.experiment.log({
                "pruning/topk": self.topk_idx})
        print(f"[INFO]: pruning/topk: {self.topk_idx}")
        # # Reset optimizer momentum for pruned channels so they can't drift back
        # optimizer = runner.optim_wrapper.optimizer
        # for group in optimizer.param_groups:
        #     for p in group['params']:
        #         if p is self._proj.gate_logits and p in optimizer.state:
        #             state = optimizer.state[p]
        #             if 'exp_avg' in state:
        #                 state['exp_avg'][self._prune_mask] = 0.0
        #             if 'exp_avg_sq' in state:
        #                 state['exp_avg_sq'][self._prune_mask] = 0.0

        # # Register grad hook to permanently zero pruned grads every backward
        # def _zero_pruned_grads(grad):
        #     grad[self._prune_mask] = 0.0
        #     return grad
        # self._proj.gate_logits.register_hook(_zero_pruned_grads)

        # self._pruned = True
        # runner.logger.info(
        #     f'[{self._tag()}] Iter {runner.iter}: pruned — '
        #     f'keeping channels {sorted(self.topk_idx)}'
        # )
        # runner.logger.info(
        #     f'[{self._tag()}] Channel order: {top_idx}'
        # )

    # def on_after_backward(self) -> None:
    #     if self._captured_conv_grad is not None:
    #         # Update EMA of saliency scores
    #         with torch.no_grad():
    #             w = self._conv.weight.detach().cpu()
    #             gw = self._captured_conv_grad.cpu()
    #             saliency = (gw * w).pow(2).sum(dim=[0, 2, 3])   # [48]
    #             self.saliency_ema = (
    #                 self.ema_momentum * self.saliency_ema
    #                 + (1 - self.ema_momentum) * saliency
    #             )
        
    #     if self.global_step % self.log_interval == 0 and self._captured_conv_grad is not None:
    #         with open(self.output_path, 'a') as f:
    #             with torch.no_grad():
    #                 # w  = self._conv.weight.detach().cpu()
    #                 # gw = self._captured_conv_grad.cpu()
    #                 # saliency_c = (gw * w).pow(2).sum(dim=[0, 2, 3])
    #                 grad_mag   = gw.pow(2).sum(dim=[0, 2, 3])
    #                 weight_mag = w.pow(2).sum(dim=[0, 2, 3])
    #                 for ch in range(len(saliency)):
    #                     f.write(
    #                         f'{self.global_step},conv,{ch},'
    #                         f'{saliency[ch].item():.6e},'
    #                         f'{self.saliency_ema[ch].item():.6e},'
    #                         f'{grad_mag[ch].item():.6e},'
    #                         f'{weight_mag[ch].item():.6e}\n'
    #                     )


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
