import json
import os
from time import time

from einops import rearrange, reduce
import torch
from ema_pytorch import EMA
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from models.utils import set_normalization, compute_stats

# Boilerplate code for training models


class Base:
    def __init__(
        self,
        *,
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader | None,
        log: str | None,
        reduce_fn: str,
        val_every: int,
        checkpoint: bool,
        norm_dict: dict[str, float | torch.Tensor] | None | str,
        max_grad_norm: float | None,
        lr_warmup: int | None,
        use_ema: bool,
        device: str,
    ):
        super().__init__()
        # Device
        self.device = device
        # Model
        self.model = model
        self.ema = (
            EMA(
                self.model,
                allow_different_devices=True,
                include_online_model=False,
            )
            if use_ema
            else None
        )
        self.use_ema = use_ema
        # Data
        self.train_dl = train_dl
        self.val_dl = val_dl
        if isinstance(norm_dict, str):
            assert norm_dict == "auto"
            norm_dict = self.estimate_stats()
        self.normalize_fn, self.denormalize_fn = set_normalization(norm_dict)
        # Optimizer
        self.lr_warmup = lr_warmup if lr_warmup is not None else 1
        # Loss
        self.red_fn = reduce_fn
        self.norm_dict = norm_dict
        self.apply_norm = norm_dict is not None
        self.max_grad_norm = (
            max_grad_norm if max_grad_norm is not None else float("inf")
        )
        # Log
        self.writer = SummaryWriter(comment=log) if log else None
        self.t_step = 0
        self.v_step = 0
        self.t_epoch = 0
        self.v_epoch = 0
        self.val_every = val_every
        self.best_val_loss = float("inf")
        self.check = checkpoint
        self.log_n_values = 3
        self.sample_train, self.sample_val = True, True
        self.training_samples = 0
        self.log_n_params(self.model)
        self.log_weights_every_epochs = 50
        self.log_grads_every_step = 1000
        self.log_audio_every_epoch = 20

    @property
    def logging(self):
        return self.writer is not None

    @property
    def validate(self):
        return self.val_dl is not None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize_fn(x)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.denormalize_fn(x)

    # Important methods to implement for subclasses ==============================================

    def models_to_device(self):
        self.model.to(self.device)
        return

    def load_checkpoint(self, path: str):
        state_d = torch.load(path, map_location="cpu")
        self.t_epoch = state_d["train_epoch"]
        self.v_epoch = state_d["val_epoch"]
        self.t_step = state_d["train_step"]
        self.v_step = state_d["val_step"]
        self.model.load_state_dict(state_d["model"])
        self.opt = state_d["optimizer"]
        norm_dict = state_d["norm_dict"]
        self.normalize_fn, self.denormalize_fn = set_normalization(norm_dict)
        return state_d

    def init_optimizer(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=1e-4)

    def init_scheduler(self):
        if self.lr_warmup > 0:
            self.sched = optim.lr_scheduler.LinearLR(
                self.opt,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.lr_warmup,
            )
        else:
            self.sched = optim.lr_scheduler.ConstantLR(
                self.opt, factor=1.0, total_iters=self.lr_warmup
            )

    def _get_x(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("Subclass must implement this method")

    def get_x(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.normalize(self._get_x(batch))

    def opt_step(self, loss: dict[str, torch.Tensor]):
        self.opt.zero_grad()
        loss["loss"].backward()
        self.clip_and_log_grads()
        self.opt.step()
        self.sched.step()
        loss["lr"] = self.sched.get_last_lr()[0]
        if self.use_ema:
            self.ema.update()
        return loss

    def log_checkpoint(self):
        if self.logging and self.check:
            logdir = os.path.join(self.writer.log_dir, "checkpoints.pt")
            state_d = {
                "train_epoch": self.t_epoch,
                "val_epoch": self.v_epoch,
                "train_step": self.t_step,
                "val_step": self.v_step,
                "model": self.model.state_dict(),
                "optimizer": self.opt,
                "norm_dict": self.norm_dict,
                "path": self.writer.log_dir,
            }
            torch.save(state_d, logdir)

    def set_train(self):
        self.model.train()
        return

    def set_eval(self):
        self.model.eval()
        if self.use_ema:
            self.ema.eval()
        return

    def log_epoch(
        self,
        *,
        batch: dict[str, torch.Tensor],
        scalars: dict[str, torch.Tensor] | None,
        prefix: str,
        step: int,
        epoch: int,
        sample: bool,
    ):
        pass

    def log_extra(self):
        return self.log_model_weights(self.model, "model")

    # ============================================================================================
    # Training methods ===========================================================================

    def _step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclass must implement this method")

    def train_step(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | float | int]:
        loss = self._step(batch)
        loss = self.opt_step(loss)
        self.training_samples += loss["batch_size"]
        return loss

    def val_step(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor | float | int]:
        return self._step(batch)

    def train_epoch(self) -> dict[str, torch.Tensor | float | int]:
        tag = "train"
        epoch_loss = 0
        self.set_train()
        start = time()
        for batch in (
            pbar := tqdm(
                self.train_dl,
                desc=f"{tag} epoch: {self.t_epoch}",
                leave=False,
            )
        ):
            train_out = self.train_step(batch)
            loss = train_out["loss"].item()
            epoch_loss += loss
            self.log_scalars(scalars=train_out, prefix=tag, step=self.t_step)
            self.t_step += 1
            self.end_of_train_step()
            pbar.set_postfix({"loss": loss})
        delta = time() - start
        extra_scalars = {
            "epoch_loss": epoch_loss / len(self.train_dl),
            "epoch": self.t_epoch,
            "training_samples (M)": self.training_samples / 1e6,
            "epoch time (m)": delta / 60,
        }
        self.log_epoch(
            batch=batch,
            scalars=extra_scalars,
            prefix=tag,
            step=self.t_step,
            epoch=self.t_epoch,
            sample=self.sample_train,
        )
        return

    def end_of_train_step(self):
        pass

    @torch.no_grad()
    def val_epoch(self) -> dict[str, torch.Tensor | float | int]:
        if self.validate and self.v_epoch % self.val_every == 0:
            tag = "val"
            epoch_loss = 0
            self.set_eval()
            start = time()
            for batch in (
                pbar := tqdm(
                    self.val_dl, desc=f"{tag} epoch: {self.v_epoch}", leave=False
                )
            ):
                val_out = self.val_step(batch)
                loss = val_out["loss"].item()
                epoch_loss += loss
                self.log_scalars(scalars=val_out, prefix=tag, step=self.v_step)
                self.v_step += 1
                pbar.set_postfix({"loss": loss})
            delta = time() - start
            extra_scalars = {
                "epoch_loss": epoch_loss / len(self.val_dl),
                "epoch": self.v_epoch,
                "epoch time (m)": delta / 60,
            }
            self.log_epoch(
                batch=batch,
                scalars=extra_scalars,
                prefix=tag,
                step=self.v_step,
                epoch=self.v_epoch,
                sample=self.sample_val,
            )
        return

    def comb_epoch(self):
        self.train_epoch()
        self.t_epoch += 1
        self.val_epoch()
        self.v_epoch += 1
        self.log_checkpoint()
        self.log_extra()
        return

    def train(self, *, epochs: int | None = 1):
        self.tot_epochs = epochs
        self.tot_iterations = self.tot_epochs * len(self.train_dl)
        self.tot_samples = self.tot_iterations * self.train_dl.batch_size
        print(
            f"Training for {self.tot_epochs} epochs / {self.tot_iterations} iterations / {self.tot_samples / 1e6:.2f}M samples"
        )
        self.models_to_device()
        self.print_data_stats()
        self.init_optimizer()
        self.init_scheduler()
        if epochs is not None:
            for _ in range(epochs):
                self.comb_epoch()
        else:
            while True:
                self.comb_epoch()

    def estimate_stats(self) -> dict[str, dict[str, float]]:
        return compute_stats(
            get_x=self._get_x,
            dl_list=[self.train_dl, self.val_dl] if self.validate else [self.train_dl],
        )

    def print_data_stats(self, n_batches: int = 16):
        batches = []
        tot_elem = 0
        for idx, batch in enumerate(
            tqdm(
                self.train_dl,
                desc=f"Printing training data stats on {n_batches} batches...",
                leave=False,
            )
        ):
            if idx > n_batches - 1:
                break
            x = self.get_x(batch)
            batch_size, *_ = x.shape
            batches.append(x)
            tot_elem += batch_size
        x = torch.cat(batches, dim=0)
        mean, std, minval, maxval = x.mean(), x.std(), x.min(), x.max()
        print(
            f"Data (estim. on {tot_elem} elements) - mean: {mean.item():.4f} | std: {std.item():.4f} | max: {minval.item():.4f} | min: {maxval.item():.4f}"
        )

    # ============================================================================================
    # Logging methods ============================================================================

    def log_model_weights(self, model: nn.Module, tag: str):
        if self.logging and self.t_epoch % self.log_weights_every_epochs == 0:
            try:
                for name, param in model.named_parameters():
                    self.writer.add_histogram(
                        f"{tag}/{name}", param, global_step=self.t_step
                    )
            except:
                pass

    def clip_and_log_grads(self):
        grads = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm,
            error_if_nonfinite=True,
            norm_type=2,
        )
        if self.logging and self.t_step % self.log_grads_every_step == 0:
            self.writer.add_scalar("grad_norm/model", grads, global_step=self.t_step)

    def log_n_params(self, model: nn.Module, tag: str = "model"):
        if self.logging:
            n_params = sum(p.numel() for p in model.parameters())
            self.writer.add_scalar(f"n_params (M)/{tag}", n_params / 1e6, global_step=0)

    def log_scalars(
        self, *, scalars: dict[str, torch.Tensor | float | int], prefix: str, step: int
    ):
        if self.logging:
            for key, value in scalars.items():
                self.writer.add_scalar(f"{prefix}/{key}", value, step)

    def log_dict(self, _dict: dict, tag: str = "constructor"):
        if self.logging:
            filtered_constructor = {
                key: value
                for key, value in _dict.items()
                if isinstance(
                    value, (int, float, str, bool, list, dict, tuple, set, type(None))
                )
            }
            try:
                self.writer.add_text(tag, str(filtered_constructor), global_step=0)
                with open(os.path.join(self.writer.log_dir, f"{tag}.json"), "w") as f:
                    json.dump(filtered_constructor, f, indent=4)
            except Exception as e:
                print(e)
                torch.save(
                    filtered_constructor, os.path.join(self.writer.log_dir, f"{tag}.pt")
                )
