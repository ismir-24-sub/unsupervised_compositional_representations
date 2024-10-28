import os
from einops import rearrange, reduce, repeat
import gin
import torch
from tqdm.auto import trange
from models.base import Base
from models import Encodec, SonyAE
import numpy as np
from museval.metrics import bss_eval
from tqdm.auto import tqdm

from models.msstft import AudioDistance, MultiScaleSTFT
from models.utils import embedding_l2_distances


class IADB(Base):
    def __init__(self, *, version: str, **kwargs):
        super().__init__(**kwargs)
        assert version in ["c", "d"]
        if version == "d":
            self.loss_fn = self.loss_d
            self.sample = self.sample_d
        else:
            self.loss_fn = self.loss_c
            self.sample = self.sample_c

    def _step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = self.get_x(batch)
        batch_size, *_ = x.shape
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        base = torch.randn_like(x)
        x_alpha = alpha * x + (1 - alpha) * base
        d = self.fwd(x_alpha=x_alpha, alpha=alpha, use_ema=False)
        loss = self.loss_fn(d=d, x=x, base=base)
        return {"loss": loss, "batch_size": batch_size}

    def fwd(
        self,
        *,
        x_alpha: torch.Tensor,
        alpha: torch.Tensor,
        use_ema: bool,
        **model_kwargs,
    ) -> dict[str, torch.Tensor | None]:
        if use_ema and self.use_ema:
            return self.ema(x_alpha, alpha.squeeze((1, 2)), **model_kwargs)
        else:
            return self.model(x_alpha, alpha.squeeze((1, 2)), **model_kwargs)

    def loss_d(
        self, *, d: torch.Tensor, x: torch.Tensor, base: torch.Tensor
    ) -> torch.Tensor:
        mse = (d - (x - base)) ** 2
        return reduce(mse, "b ... -> b", self.red_fn).mean()

    @torch.no_grad()
    def sample_d(self, *, base: torch.Tensor, steps: int = 50, **model_kwargs):
        self.set_eval()
        if self.use_ema:
            self.ema.to(self.device)
        batch_size, *_ = base.shape
        dt = 1 / steps
        noises = trange(steps, desc="sampling...", leave=False)
        for t in noises:
            alpha = repeat(
                torch.tensor(t / steps, device=self.device), " -> b 1 1", b=batch_size
            )
            d = self.fwd(x_alpha=base, alpha=alpha, use_ema=True, **model_kwargs)
            base = base + dt * d
            noises.set_postfix(
                {"t": t, ": min": base.min().item(), "max": base.max().item()}
            )
        if self.use_ema:
            self.ema.restore_ema_model_device()
        return base

    def loss_c(
        self, *, d: torch.Tensor, x: torch.Tensor, base: torch.Tensor
    ) -> torch.Tensor:
        mse = (d - x) ** 2
        return reduce(mse, "b ... -> b", self.red_fn).mean()

    @torch.no_grad()
    def sample_c(self, *, base: torch.Tensor, steps: int = 50, **model_kwargs):
        self.set_eval()
        if self.use_ema:
            self.ema.to(self.device)
        batch_size, *_ = base.shape
        noises = trange(steps, desc="sampling...", leave=False)
        for t in noises:
            alpha = repeat(
                torch.tensor(t / steps, device=self.device), " -> b 1 1", b=batch_size
            )
            d = self.fwd(x_alpha=base, alpha=alpha, use_ema=True, **model_kwargs)
            dt = (1 - ((t + 1) / steps)) / (1 - (t / steps))
            base = d + dt * (base - d)
            noises.set_postfix(
                {"t": t, ": min": base.min().item(), "max": base.max().item()}
            )
        if self.use_ema:
            self.ema.restore_ema_model_device()
        return base


class AudioIADB(IADB):

    @property
    def sr(self):
        raise NotImplementedError("Subclass must implement this property")

    @torch.no_grad()
    def to_audio(self, x: torch.Tensor):
        raise NotImplementedError("Subclass must implement this method")

    @torch.no_grad()
    def x_to_audio(self, x: torch.Tensor):
        return self.to_audio(self.denormalize(x))
    
    @torch.no_grad()
    def log_epoch(
        self,
        *,
        batch: dict[str, torch.Tensor],
        scalars: dict[str, torch.Tensor] | None,
        prefix: str,
        step: int,
        epoch: int,
        sample: bool = True,
    ):
        if self.logging:
            if scalars is not None:
                self.log_scalars(scalars=scalars, prefix=prefix, step=step)
            x = self.get_x(batch)
            x = x[: self.log_n_values]
            bs, *_ = x.shape
            self.set_eval()
            if epoch == 0:
                x_audio = self.x_to_audio(x)
                for idx in range(bs):
                    self.writer.add_audio(
                        f"{prefix}_real/{idx}",
                        x_audio[idx].squeeze(),
                        step,
                        sample_rate=self.sr,
                    )
            if sample:
                # Generate
                x_gen = self.sample(base=torch.randn_like(x))
                x_gen_audio = self.x_to_audio(x_gen)
                # Log histogram
                self.writer.add_histogram(f"{prefix}/real", x, step)
                self.writer.add_histogram(f"{prefix}/gen", x_gen, step)
                # Log audio
                for idx in range(bs):
                    self.writer.add_audio(
                        f"{prefix}_gen/{idx}",
                        x_gen_audio[idx].squeeze(),
                        step,
                        sample_rate=self.sr,
                    )


@gin.configurable
class LatentIADB(AudioIADB):

    def __init__(self, *, encodec: Encodec | SonyAE, **iadb_kwargs):
        super().__init__(**iadb_kwargs)
        self.encodec = encodec
        self.encodec.eval()

    def models_to_device(self):
        self.model.to(self.device)
        self.encodec.to(self.device)
        return

    @property
    def sr(self):
        return self.encodec.sr

    @torch.no_grad()
    def to_audio(self, x: torch.Tensor):
        return self.encodec.decode(x)

    def set_train(self):
        self.model.train()
        self.encodec.eval()

    def set_eval(self):
        self.model.eval()
        self.encodec.eval()


class AudioDiffAE(AudioIADB):

    def __init__(self, *, encoder: torch.nn.Module, **iadb_kwargs):
        super().__init__(**iadb_kwargs)
        self.encoder = encoder
        self.log_n_params(self.encoder, "encoder")
        ms_stft = MultiScaleSTFT(
            scales=[2048, 1024, 512, 256, 128, 64],
            sample_rate=self.sr,
        )
        self.audio_distance = AudioDistance(ms_stft)

    def models_to_device(self):
        super().models_to_device()
        self.encoder.to(self.device)
        self.audio_distance.to(self.device)
        return

    def init_optimizer(self):
        self.opt = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.encoder.parameters()),
            lr=1e-4,
        )

    def load_checkpoint(self, path: str):
        state_d = super().load_checkpoint(path)
        self.encoder.load_state_dict(state_d["encoder"])
        return state_d

    def _step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = self.get_x(batch)
        batch_size, *_ = x.shape
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        base = torch.randn_like(x)
        cond = self.encoder(x)
        x_alpha = alpha * x + (1 - alpha) * base
        d = self.fwd(x_alpha=x_alpha, alpha=alpha, use_ema=False, cond=cond)
        loss = self.loss_fn(d=d, x=x, base=base)
        return {"loss": loss, "batch_size": batch_size}

    @torch.no_grad()
    def log_epoch(
        self,
        *,
        batch: dict[str, torch.Tensor],
        scalars: dict[str, torch.Tensor] | None,
        prefix: str,
        step: int,
        epoch: int,
        sample: bool = True,
    ):
        if self.logging:
            if scalars is not None:
                self.log_scalars(scalars=scalars, prefix=prefix, step=step)
            if sample:
                self.set_eval()
                x = self.get_x(batch)
                x_audio = self.x_to_audio(x)
                # Encode to semantic vector
                cond = self.encoder(x)
                # Reconstruct
                x_gen = self.sample(base=torch.randn_like(x), cond=cond)
                # Log audio distance
                x_gen_audio = self.x_to_audio(x_gen)
                audio_distance = self.audio_distance(x=x_gen_audio, y=x_audio)
                self.log_scalars(scalars=audio_distance, prefix=prefix, step=step)
                # Log histograms
                self.writer.add_histogram(f"{prefix}/real", x, step)
                self.writer.add_histogram(f"{prefix}/gen", x_gen, step)
                self.writer.add_histogram(f"{prefix}/cond", cond, step)
                if epoch % self.log_audio_every_epoch == 0:
                    # Log audio
                    for idx in range(self.log_n_values):
                        self.writer.add_audio(
                            f"{prefix}_gen/{idx}",
                            x_gen_audio[idx].squeeze(),
                            step,
                            sample_rate=self.sr,
                        )
                        self.writer.add_audio(
                            f"{prefix}_real/{idx}",
                            x_audio[idx].squeeze(),
                            step,
                            sample_rate=self.sr,
                        )

    def clip_and_log_grads(self):
        grads_model = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.max_grad_norm,
            error_if_nonfinite=True,
        )
        grads_encoder = torch.nn.utils.clip_grad_norm_(
            self.encoder.parameters(),
            self.max_grad_norm,
            error_if_nonfinite=True,
        )
        if self.logging and self.t_step % self.log_grads_every_step == 0:
            self.writer.add_scalar(
                "grad_norm/model",
                grads_model,
                global_step=self.t_step,
            )
            self.writer.add_scalar(
                "grad_norm/encoder",
                grads_encoder,
                global_step=self.t_step,
            )

    def set_train(self):
        self.model.train()
        self.encoder.train()
        return

    def set_eval(self):
        self.model.eval()
        self.encoder.eval()
        return

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
                "encoder": self.encoder.state_dict(),
                "norm_dict": self.norm_dict,
                "path": self.writer.log_dir,
            }
            torch.save(state_d, logdir)

    def log_extra(self):
        try:
            self.log_model_weights(self.model, "model")
            self.log_model_weights(self.encoder, "encoder")
        except Exception as e:
            print(f"Error logging extra: {e}")
            pass


# Decomposition =========================================================================================================


class AudioDecompIADB(AudioDiffAE):

    def __init__(self, sources: list[str], **iadb_kwargs):
        super().__init__(**iadb_kwargs)
        self.sources = sources
        self.log_separation_metrics_every_epoch = 10
        self.log_full_separation_metrics_every_epoch = 25
        self.n_batches_full_separation = 8

    @property
    def n_sources(self):
        return len(self.sources)

    @torch.no_grad()
    def component_to_audio(self, x: torch.Tensor):
        raise NotImplementedError("Subclass must implement this method")

    def fwd(
        self,
        *,
        x_alpha: torch.Tensor,
        alpha: torch.Tensor,
        use_ema: bool,
        **model_kwargs,
    ) -> dict[str, torch.Tensor | None]:
        if use_ema and self.use_ema:
            return self.ema(x_alpha, alpha.squeeze((1, 2)), **model_kwargs)["out"]
        else:
            return self.model(x_alpha, alpha.squeeze((1, 2)), **model_kwargs)["out"]

    @torch.no_grad()
    def log_epoch(
        self,
        *,
        batch: dict[str, torch.Tensor],
        scalars: dict[str, torch.Tensor] | None,
        prefix: str,
        step: int,
        epoch: int,
        sample: bool = True,
    ):
        if self.logging:
            if scalars is not None:
                self.log_scalars(scalars=scalars, prefix=prefix, step=step)
            if sample:
                self.set_eval()
                x = self.get_x(batch)
                # Log real and reconstructed audio
                x_audio = self.x_to_audio(x)
                # Extract components and reconstruct
                cond = self.encoder(x)
                # Measure l2 distance between components
                if isinstance(cond, torch.Tensor):
                    distances = embedding_l2_distances(cond)
                    self.writer.add_scalar(
                        f"{prefix}/embeddings_l2_distance", distances, step
                    )
                x_gen = self.sample(base=torch.randn_like(x), cond=cond)
                x_gen_audio = self.x_to_audio(x_gen)
                # Log audio distance
                audio_distance = self.audio_distance(x=x_gen_audio, y=x_audio)
                self.log_scalars(scalars=audio_distance, prefix=prefix, step=step)
                # Log MSE between x and reconstruction
                mse = reduce((x - x_gen) ** 2, "b ... -> b", "sum").mean()
                self.writer.add_scalar(f"{prefix}/spec_mse", mse, step)
                # Log audio
                if epoch % self.log_audio_every_epoch == 0:
                    x = x[: self.log_n_values]
                    x_gen = x_gen[: self.log_n_values]
                    x_audio = x_audio[: self.log_n_values]
                    x_gen_audio = x_gen_audio[: self.log_n_values]
                    if isinstance(cond, list):
                        cond = [c[: self.log_n_values] for c in cond]
                    else:
                        cond = cond[: self.log_n_values]
                    for idx in range(self.log_n_values):
                        self.writer.add_audio(
                            f"{prefix}_real/{idx}",
                            x_audio[idx].squeeze(),
                            step,
                            sample_rate=self.sr,
                        )
                        self.writer.add_audio(
                            f"{prefix}_gen/{idx}",
                            x_gen_audio[idx].squeeze(),
                            step,
                            sample_rate=self.sr,
                        )
                    # Generate each component
                    for nc in range(self.n_sources):
                        x_gen_cond = self.sample(
                            base=torch.randn_like(x), cond=cond, component_idx=nc
                        )
                        x_gen_cond_audio = self.component_to_audio(x_gen_cond)
                        for idx in range(self.log_n_values):
                            self.writer.add_audio(
                                f"{prefix}_gen_cond_{nc}/{idx}",
                                x_gen_cond_audio[idx].squeeze(),
                                step,
                                sample_rate=self.sr,
                            )
                    # Logging real sources
                    for k, v in batch.items():
                        if k in ["drums", "bass", "piano"]:
                            v = v[: self.log_n_values].to(self.device)
                            v_audio = self.to_audio(v)
                            for idx in range(len(v)):
                                self.writer.add_audio(
                                    f"{prefix}_{k}/{idx}",
                                    v_audio[idx].squeeze(),
                                    step,
                                    sample_rate=self.sr,
                                )
                # Log histograms
                self.writer.add_histogram(f"{prefix}/real", x, step)
                self.writer.add_histogram(f"{prefix}/gen", x_gen, step)
                if isinstance(cond, torch.Tensor):
                    self.writer.add_histogram(f"{prefix}/cond", cond, step)
                # Log separation metrics
                self.log_separation_metrics(batch, prefix, step, epoch)
                # Log extra stuff (like spectrograms)
                self.log_extra_variables(batch, prefix, step, epoch)

    @torch.no_grad()
    def log_separation_metrics(
        self, batch: dict[str, torch.Tensor], prefix: str, step: int, epoch: int
    ):
        if (
            self.logging
            and epoch > 0
            and epoch % self.log_separation_metrics_every_epoch == 0
        ):
            self.set_eval()
            sdrs, sirs, sars = self.separate(batch)
            if sdrs is None:
                return
            for n in range(self.n_sources):
                self.writer.add_scalar(f"{prefix}/sdr_source_{n}", sdrs[n], step)
                self.writer.add_scalar(f"{prefix}/sir_source_{n}", sirs[n], step)
                self.writer.add_scalar(f"{prefix}/sar_source_{n}", sars[n], step)
            mean_sdr = reduce(sdrs, "n -> ()", "mean")
            mean_sir = reduce(sirs, "n -> ()", "mean")
            mean_sar = reduce(sars, "n -> ()", "mean")
            self.writer.add_scalar(f"{prefix}/mean_sdr", mean_sdr, step)
            self.writer.add_scalar(f"{prefix}/mean_sir", mean_sir, step)
            self.writer.add_scalar(f"{prefix}/mean_sar", mean_sar, step)

    @torch.no_grad()
    def separate(self, batch: dict[str, torch.Tensor]):
        x = self.get_x(batch)
        bs, *_ = x.shape
        audio_sources_real = [
            self.to_audio(batch[s].to(self.device)) for s in self.sources
        ]
        cond = self.encoder(x)
        base = torch.randn_like(x)
        audio_sources_gen = [
            self.component_to_audio(self.sample(base=base, cond=cond, component_idx=i))
            for i in range(self.n_sources)
        ]
        sdrs, sirs, sars = [], [], []
        swap_tc = lambda x: rearrange(x, "c t -> t c")
        for i in trange(
            bs,
            desc="computing separation metrics on batch",
            leave=False,
        ):
            real = torch.stack([swap_tc(a[i]) for a in audio_sources_real], dim=0)
            est = torch.stack([swap_tc(a[i]) for a in audio_sources_gen], dim=0)
            try:
                sdr, _, sir, sar, _ = bss_eval(
                    real.cpu().numpy(),
                    est.cpu().numpy(),
                    window=3.5 * self.sr,
                    hop=1.5 * self.sr,
                    compute_permutation=True,
                )
                sdr = reduce(sdr, "n w -> n", "mean")
                sir = reduce(sir, "n w -> n", "mean")
                sar = reduce(sar, "n w -> n", "mean")
                sdrs.append(sdr)
                sirs.append(sir)
                sars.append(sar)
            except Exception:
                pass
        if len(sdrs) == 0:
            return None, None, None
        sdrs = np.stack(sdrs)
        sirs = np.stack(sirs)
        sars = np.stack(sars)
        sdrs = reduce(sdrs, "b n -> n", "mean")
        sirs = reduce(sirs, "b n -> n", "mean")
        sars = reduce(sars, "b n -> n", "mean")
        return sdrs, sirs, sars

    @torch.no_grad()
    def log_extra_variables(
        self,
        batch: dict[str, torch.Tensor],
        prefix: str,
        step: int,
        epoch: int,
    ):
        pass

    @torch.no_grad()
    def separation_epoch(self):
        if (
            self.logging
            and self.t_epoch > 0
            and self.t_epoch % self.log_full_separation_metrics_every_epoch == 0
        ):
            self.set_eval()
            dl = self.val_dl if self.validate else self.train_dl
            sdrs, sirs, sars = [], [], []
            for i, batch in enumerate(
                tqdm(
                    dl,
                    desc=f"separation metrics on {self.n_batches_full_separation} batches",
                    leave=False,
                )
            ):
                if i > (self.n_batches_full_separation - 1):
                    break
                sdr, sir, sar = self.separate(batch)
                if sdr is None:
                    continue
                sdr = rearrange(sdr, "n -> 1 n")
                sir = rearrange(sir, "n -> 1 n")
                sar = rearrange(sar, "n -> 1 n")
                sdrs.append(sdr)
                sirs.append(sir)
                sars.append(sar)
            sdrs = np.concatenate(sdrs, axis=0)
            sirs = np.concatenate(sirs, axis=0)
            sars = np.concatenate(sars, axis=0)
            mean_sdr = sdrs.mean(0)  # (b, n) -> (n)
            mean_sir = sirs.mean(0)  # (b, n) -> (n)
            mean_sar = sars.mean(0)  # (b, n) -> (n)
            std_sdr = sdrs.std(0)  # (b, n) -> (n)
            std_sir = sirs.std(0)  # (b, n) -> (n)
            std_sar = sars.std(0)  # (b, n) -> (n)
            mean_mean_sdr = mean_sdr.mean()
            mean_mean_sir = mean_sir.mean()
            mean_mean_sar = mean_sar.mean()
            for i in range(self.n_sources):
                self.writer.add_scalar(
                    f"separation/sdr_source_{i}",
                    mean_sdr[i],
                    self.t_step,
                )
                self.writer.add_scalar(
                    f"separation/std_sdr_source_{i}",
                    std_sdr[i],
                    self.t_step,
                )
                self.writer.add_scalar(
                    f"separation/sir_source_{i}",
                    mean_sir[i],
                    self.t_step,
                )
                self.writer.add_scalar(
                    f"separation/std_sir_source_{i}",
                    std_sir[i],
                    self.t_step,
                )
                self.writer.add_scalar(
                    f"separation/sar_source_{i}",
                    mean_sar[i],
                    self.t_step,
                )
                self.writer.add_scalar(
                    f"separation/std_sar_source_{i}",
                    std_sar[i],
                    self.t_step,
                )
            self.writer.add_scalar("separation/mean_sdr", mean_mean_sdr, self.t_step)
            self.writer.add_scalar("separation/mean_sir", mean_mean_sir, self.t_step)
            self.writer.add_scalar("separation/mean_sar", mean_mean_sar, self.t_step)

    def comb_epoch(self):
        self.train_epoch()
        self.t_epoch += 1
        self.val_epoch()
        self.v_epoch += 1
        self.separation_epoch()
        self.log_checkpoint()
        self.log_extra()
        return


@gin.configurable
class DecompLatentIADB(AudioDecompIADB, LatentIADB):

    def models_to_device(self):
        super().models_to_device()
        self.encodec.to(self.device)
        return

    def set_train(self):
        super().set_train()
        self.encodec.eval()

    def set_eval(self):
        super().set_eval()
        self.encodec.eval()

    @torch.no_grad()
    def component_to_audio(self, x: torch.Tensor):
        return self.x_to_audio(x)

    def get_components(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        sources = [batch[source] for source in self.sources]
        return torch.stack(sources, dim=1).to(self.device)

    def _get_x(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["data"].to(self.device)


@gin.configurable
class MaskingLatentIADB(AudioIADB):

    def __init__(self, 
                 pre_train_decomp: DecompLatentIADB, 
                 sources: list[str], 
                 p_drop: float, 
                 add_mask_cond: bool, 
                 **kwargs):
        super().__init__(**kwargs)
        self.pre_train_decomp = pre_train_decomp
        self.pre_train_decomp.set_eval()
        self.sources = sources
        self.p_drop = p_drop
        self.add_mask_cond = add_mask_cond
        self.masks = self.init_masks()
        self.probs = self.init_probs()

    @property
    def sr(self):
        return self.pre_train_decomp.sr

    def set_train(self):
        super().set_train()
        self.pre_train_decomp.set_eval()

    def set_eval(self):
        super().set_eval()
        self.pre_train_decomp.set_eval()

    def models_to_device(self):
        super().models_to_device()
        self.pre_train_decomp.models_to_device()
        return
    
    @torch.no_grad()
    def init_masks(self):
        x = self.pre_train_decomp.get_x(next(iter(self.train_dl))).cpu()
        x = self.pre_train_decomp.encoder(x)
        _, n, c, t = x.shape
        assert n == self.n_sources, f"error. n = {n}, n_sources = {self.n_sources}"
        m = torch.zeros(1, self.n_sources, c, t).float()
        masks = []
        for i in range(1, 2**self.n_sources):
            to_be_masked = [int(j) for j in f"{i:0{self.n_sources}b}"]
            m_i = m.clone()
            for j in range(self.n_sources):
                if to_be_masked[j]:
                    m_i[:, j] = 1
            masks.append(m_i)
        return masks # list of 2**n_sources - 1 with shape (1, n_sources * c, t)
    
    def init_probs(self):
        drop_all = self.p_drop
        n_comp_masks = 2**self.n_sources - 1 - 1 # extra -1 for the all ones mask
        drop_per_mask = (1 - drop_all) / n_comp_masks
        probs = [drop_per_mask] * n_comp_masks + [drop_all]
        assert sum(probs) == 1, f"error. sum(probs) = {sum(probs)}"

    @property
    def n_sources(self):
        return len(self.sources)
    
    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b n c t -> b (n c) t")
    
    def _deshape(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b (n c) t -> b n c t", n=self.n_sources)
    
    def _get_x(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.pre_train_decomp.get_x(batch)
        with torch.no_grad():
            x = self.pre_train_decomp.encoder(x)  # b n c t
        return self._reshape(x)
    
    def get_components(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        sources = [batch[source] for source in self.sources]
        return torch.stack(sources, dim=1).to(self.device)

    def sample_mask(self, batch_size: int):
        rand_mask = np.random.choice(range(len(self.masks)), size=(batch_size,), replace=True, p=self.probs)
        masks = [self.masks[i] for i in rand_mask]
        masks = torch.cat(masks, dim=0).to(self.device)
        return masks

    def apply_mask(self, *, x_t: torch.Tensor, x: torch.Tensor, mask: torch.Tensor):
        x_t = self._deshape(x_t)
        x = self._deshape(x)
        x_t = x_t * mask + x * (1 - mask)
        return self._reshape(x_t)

    def sample_and_apply_mask(self, *, x_t: torch.Tensor, x: torch.Tensor):
        b, *_ = x.shape
        mask = self.sample_mask(b)
        return self.apply_mask(x_t=x_t, x=x, mask=mask), mask

    def _step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = self.get_x(batch)
        batch_size, *_ = x.shape
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        base = torch.randn_like(x)
        x_alpha = alpha * x + (1 - alpha) * base
        x_alpha, mask = self.sample_and_apply_mask(x_t=x_alpha, x=x)
        cond = self._reshape(mask) if self.add_mask_cond else None
        d = self.fwd(x_alpha=x_alpha, alpha=alpha, use_ema=False, cond=cond)
        loss = self.loss_fn(d=d, x=x, base=base)
        return {"loss": loss, "batch_size": batch_size}

    @torch.no_grad()
    def to_audio(self, x: torch.Tensor):
        x = self._deshape(x)  # b n c t
        b, _, _, t = x.shape
        x = self.pre_train_decomp.sample(base=torch.randn(b, 128, t, device=self.device), cond=x)
        return self.pre_train_decomp.to_audio(x)
    
    @torch.no_grad()
    def component_to_audio(self, x: torch.Tensor, c_idx: int):
        x = self._deshape(x)  # b n c t
        b, _, _, t = x.shape
        x = self.pre_train_decomp.sample(base=torch.randn(b, 128, t, device=self.device), cond=x, component_idx=c_idx)
        return self.pre_train_decomp.to_audio(x)
    
    @torch.no_grad()
    def log_epoch(
        self,
        *,
        batch: dict[str, torch.Tensor],
        scalars: dict[str, torch.Tensor] | None,
        prefix: str,
        step: int,
        epoch: int,
        sample: bool = True,
    ):
        if self.logging:
            if scalars is not None:
                self.log_scalars(scalars=scalars, prefix=prefix, step=step)
            if sample and epoch % self.log_audio_every_epoch == 0:
                self.set_eval()
                x = self.get_x(batch)  # b (n c) t
                x = x[: self.log_n_values]
                uncond_mask = repeat(self.masks[-1].to(self.device), "1 n c t -> b n c t", b=self.log_n_values)
                base = torch.randn_like(x)
                uncond_gen = self.sample(base=base, cond=self._reshape(uncond_mask) if self.add_mask_cond else None)
                uncond_gen_audio = self.to_audio(uncond_gen)
                x_audio = self.to_audio(x)
                for i in range(self.log_n_values):
                    self.writer.add_audio(
                        f"{prefix}_real/{i}",
                        x_audio[i].squeeze(),
                        step,
                        sample_rate=self.sr,
                    )
                    self.writer.add_audio(
                        f"{prefix}_gen_uncond/{i}",
                        uncond_gen_audio[i].squeeze(),
                        step,
                        sample_rate=self.sr,
                    )
                for comp_idx in range(len(self.masks) - 1):
                    mask = repeat(self.masks[comp_idx].to(self.device), "1 n c t -> b n c t", b=self.log_n_values)
                    _base = self.apply_mask(x_t=base, x=x, mask=mask)
                    cond_gen = self.sample(base=_base, cond=self._reshape(mask) if self.add_mask_cond else None)
                    cond_gen_audio = self.to_audio(cond_gen)
                    for i in range(self.log_n_values):
                        self.writer.add_audio(
                            f"{prefix}_gen_cond_{comp_idx}/{i}",
                            cond_gen_audio[i].squeeze(),
                            step,
                            sample_rate=self.sr,
                        )
