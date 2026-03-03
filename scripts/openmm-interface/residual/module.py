# Copyright Universitat Pompeu Fabra 2020-2023  https://www.compscience.org
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

from collections import defaultdict
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.nn.functional import local_response_norm, mse_loss, l1_loss
from torch import Tensor
from typing import Optional, Dict, Tuple

from lightning import LightningModule
from residual.models.model import create_model, load_model
from residual.models.utils import dtype_mapping
import torch_geometric.transforms as T


class FloatCastDatasetWrapper(T.BaseTransform):
    """A transform that casts all floating point tensors to a given dtype.
    tensors to a given dtype.
    """

    def __init__(self, dtype=torch.float64):
        super(FloatCastDatasetWrapper, self).__init__()
        self._dtype = dtype

    def forward(self, data):
        for key, value in data:
            if torch.is_tensor(value) and torch.is_floating_point(value):
                setattr(data, key, value.to(self._dtype))
        return data


class EnergyRefRemover(T.BaseTransform):
    """A transform that removes the atom reference energy from the energy of a
    dataset.
    """

    def __init__(self, atomref):
        super(EnergyRefRemover, self).__init__()
        self._atomref = atomref

    def forward(self, data):
        self._atomref = self._atomref.to(data.z.device).type(data.y.dtype)
        if "y" in data:
            data.y.index_add_(0, data.batch, -self._atomref[data.z])
        return data


class LNNP(LightningModule):

    def __init__(self, hparams, prior_model=None, mean=None, std=None):
        super(LNNP, self).__init__()
        if "charge" not in hparams:
            hparams["charge"] = False
        if "spin" not in hparams:
            hparams["spin"] = False

        self.save_hyperparameters(hparams)

        if self.hparams.load_model:
            self.model = load_model(self.hparams.load_model, args=self.hparams)
        elif self.hparams.pretrained_model:
            self.model = load_model(self.hparams.pretrained_model, args=self.hparams, mean=mean, std=std, prior_model=prior_model)
        else:
            self.model = create_model(self.hparams, prior_model, mean, std)

        # initialize exponential smoothing
        self.ema = None
        self._reset_ema_dict()

        # initialize loss collection
        self.losses = None
        self._reset_losses_dict()

        self.data_transform = FloatCastDatasetWrapper(
            dtype_mapping[self.hparams.precision]
        )
        if self.hparams.remove_ref_energy:
            self.data_transform = T.Compose(
                [
                    EnergyRefRemover(self.model.prior_model[-1].initial_atomref),
                    self.data_transform,
                ]
            )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.lr_schedule == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, self.hparams.lr_cosine_length)
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        elif self.hparams.lr_schedule == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                factor=self.hparams.lr_factor,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": getattr(self.hparams, "lr_metric", "val_loss"),
                "interval": "epoch",
                "frequency": 1,
            }

        return [optimizer], [lr_scheduler]

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        box: Optional[Tensor] = None,
        q: Optional[Tensor] = None,
        s: Optional[Tensor] = None,
        extra_args: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        return self.model(z, pos, batch=batch, box=box, q=q, s=s, extra_args=extra_args)

    def training_step(self, batch, batch_idx):
        return self.step(batch, [mse_loss], "train")

    def validation_step(self, batch, batch_idx, *args):

        is_val = len(args) == 0 or (len(args) > 0 and args[0] == 0)
        if is_val:
            step_type = {"loss_fn_list": [l1_loss, mse_loss], "stage": "val"}
        else:
            step_type = {"loss_fn_list": [l1_loss], "stage": "test"}
        return self.step(batch, **step_type)

    def test_step(self, batch, batch_idx):
        return self.step(batch, [l1_loss], "test")

    def _compute_losses(self, y, neg_y, noise_pred, batch, loss_fn, stage):

        loss_y, loss_neg_y, loss_noise= torch.tensor(0.0, device=self.device), torch.tensor(
            0.0, device=self.device), torch.tensor(0.0, device=self.device,
        )

        loss_name = loss_fn.__name__
        if self.hparams.derivative and "neg_dy" in batch:
            loss_neg_y = loss_fn(neg_y, batch.neg_dy)
            loss_neg_y = self._update_loss_with_ema(
                stage, "neg_dy", loss_name, loss_neg_y
            )
        if "y" in batch:
            loss_y = loss_fn(y, batch.y)
            loss_y = self._update_loss_with_ema(stage, "y", loss_name, loss_y)
         # Compute pretraining (denoising) loss
        if self.hparams.denoising_weight > 0 and "pos_target" in batch:
            normalized_pos_target = self.model.pos_normalizer(batch.pos_target)
            loss_noise = loss_fn(noise_pred, normalized_pos_target)
        return {"y": loss_y, "neg_dy": loss_neg_y, "noise": loss_noise}

    def _update_loss_with_ema(self, stage, type, loss_name, loss):

        alpha = getattr(self.hparams, f"ema_alpha_{type}")
        if stage in ["train", "val"] and alpha < 1:
            ema = (
                self.ema[stage][type][loss_name]
                if loss_name in self.ema[stage][type]
                else loss.detach()
            )
            loss = alpha * loss + (1 - alpha) * ema
            self.ema[stage][type][loss_name] = loss.detach()
        return loss

    def step(self, batch, loss_fn_list, stage):

        assert len(loss_fn_list) > 0
        assert self.losses is not None
        batch = self.data_transform(batch)
        with torch.set_grad_enabled(stage == "train" or self.hparams.derivative):
            extra_args = batch.to_dict()
            for a in ("y", "neg_dy", "z", "pos", "batch", "box", "q", "s"):
                if a in extra_args:
                    del extra_args[a]
            # TODO: the model doesn't necessarily need to return a derivative once
            # Union typing works under TorchScript (https://github.com/pytorch/pytorch/pull/53180)
            y, neg_dy, noise_pred = self(
                batch.z,
                batch.pos,
                batch=batch.batch,
                box=batch.box if "box" in batch else None,
                q=batch.q if self.hparams.charge else None,
                s=batch.s if self.hparams.spin else None,
                extra_args=extra_args,
            )
        denoising_is_on = ("pos_target" in batch) and (self.hparams.denoising_weight > 0)

        if self.hparams.derivative and "y" not in batch:
            # "use" both outputs of the model's forward function but discard the first
            # to only use the negative derivative and avoid 'Expected to have finished reduction
            # in the prior iteration before starting a new one.', which otherwise get's
            # thrown because of setting 'find_unused_parameters=False' in the DDPPlugin
            neg_dy = neg_dy + y.sum() * 0
        if "y" in batch and batch.y.ndim == 1:
            batch.y = batch.y.unsqueeze(1)
            if (noise_pred is not None) and not denoising_is_on:
                # "use" both outputs of the model's forward (see comment above).
                y = y + noise_pred.sum() * 0 
        if denoising_is_on and "y" not in batch:
            noise_pred = noise_pred + y.sum() * 0           
        for loss_fn in loss_fn_list:
            step_losses = self._compute_losses(y, neg_dy, noise_pred, batch, loss_fn, stage)

            loss_name = loss_fn.__name__
            if self.hparams.neg_dy_weight > 0:
                self.losses[stage]["neg_dy"][loss_name].append(
                    step_losses["neg_dy"].detach()
                )
            if self.hparams.y_weight > 0:
                self.losses[stage]["y"][loss_name].append(step_losses["y"].detach())
            if self.hparams.denoising_weight > 0:
                self.losses[stage]["noise"][loss_name].append(step_losses["noise"].detach())
            total_loss = (
                step_losses["y"] * self.hparams.y_weight
                + step_losses["neg_dy"] * self.hparams.neg_dy_weight
                + step_losses['noise'] * self.hparams.denoising_weight
            )
            self.losses[stage]["total"][loss_name].append(total_loss.detach())
        # Frequent per-batch logging for training
        if stage == 'train':
            # Extracting the latest (most recent) loss values for each loss type for the 'train' stage
            train_metrics = {
                'train_y_per_step': self.losses['train']['y'][loss_name][-1].item() if 'y' in self.losses['train'] and len(self.losses['train']['y'][loss_name]) > 0 else 0,
                'train_neg_dy_per_step': self.losses['train']['neg_dy'][loss_name][-1].item() if 'neg_dy' in self.losses['train'] and len(self.losses['train']['neg_dy'][loss_name]) > 0 else 0,
                'train_noise_per_step': self.losses['train']['noise'][loss_name][-1].item() if 'noise' in self.losses['train'] and len(self.losses['train']['noise'][loss_name]) > 0 else 0,
                'lr_per_step': self.trainer.optimizers[0].param_groups[0]["lr"],
                'step': self.trainer.global_step,
                'batch_pos_mean': batch.pos.mean().item(),
            }
       
            self.log_dict(train_metrics, sync_dist=True, prog_bar=True, logger=True)
        return total_loss

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs["optimizer"] if "optimizer" in kwargs else args[2]
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(self.trainer.global_step + 1)
                / float(self.hparams.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def _get_mean_loss_dict_for_type(self, type):
        assert type in ['y', 'neg_dy', 'noise', 'total'], f"Unexpected loss type: {type}"
        assert self.losses is not None
        mean_losses = {}
        for stage in ["train", "val", "test"]:
            if type not in self.losses[stage]:  # If no losses of this type were recorded
                continue  # Skip this type
            for loss_fn_name in self.losses[stage][type].keys():
                if len(self.losses[stage][type][loss_fn_name]) > 0:
                    mean_losses[stage + "_" + type + "_" + loss_fn_name] = torch.stack(
                        self.losses[stage][type][loss_fn_name]
                    ).mean()
        return mean_losses

    def on_validation_epoch_end(self):
        if not self.trainer.sanity_checking:
            # construct dict of logged metrics
            result_dict = {
                "epoch": float(self.current_epoch),
                "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            }
            result_dict.update(self._get_mean_loss_dict_for_type("total"))
            result_dict.update(self._get_mean_loss_dict_for_type("y"))
            result_dict.update(self._get_mean_loss_dict_for_type("neg_dy"))
            result_dict.update(self._get_mean_loss_dict_for_type("noise"))

            self.log_dict(result_dict, sync_dist=True)

        self._reset_losses_dict()

    def on_test_epoch_end(self):
        # Log all test losses
        if not self.trainer.sanity_checking:
            result_dict = {}
            result_dict.update(self._get_mean_loss_dict_for_type("total"))
            result_dict.update(self._get_mean_loss_dict_for_type("y"))
            result_dict.update(self._get_mean_loss_dict_for_type("neg_dy"))
            result_dict.update(self._get_mean_loss_dict_for_type("noise"))
            # Get only test entries
            result_dict = {k: v for k, v in result_dict.items() if k.startswith("test")}
            self.log_dict(result_dict, sync_dist=True)

    def _reset_losses_dict(self):
        # Losses has an entry for each stage in ["train", "val", "test"]
        # Each entry has an entry with "total", "y" and "neg_dy"
        # Each of these entries has an entry for each loss_fn (e.g. mse_loss)
        # The loss_fn values are not known in advance
        self.losses = {}
        for stage in ["train", "val", "test"]:
            self.losses[stage] = {}
            for loss_type in ["total", "y", "neg_dy", "noise"]:
                self.losses[stage][loss_type] = defaultdict(list)

    def _reset_ema_dict(self):
        self.ema = {}
        for stage in ["train", "val"]:
            self.ema[stage] = {}
            for loss_type in ["y", "neg_dy", "noise"]:
                self.ema[stage][loss_type] = {}
