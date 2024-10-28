# Credits https://github.com/acids-ircam/RAVE

from typing import Sequence

import librosa as li
import torch
import torch.nn as nn
import torchaudio
from einops import rearrange


def mean_difference(
    target: torch.Tensor,
    value: torch.Tensor,
    norm: str = "L1",
    relative: bool = False,
):
    diff = target - value
    if norm == "L1":
        diff = diff.abs().mean()
        if relative:
            diff = diff / target.abs().mean()
        return diff
    elif norm == "L2":
        diff = (diff * diff).mean()
        if relative:
            diff = diff / (target * target).mean()
        return diff
    else:
        raise Exception(f"Norm must be either L1 or L2, got {norm}")


class MelScale(nn.Module):

    def __init__(self, sample_rate: int, n_fft: int, n_mels: int) -> None:
        super().__init__()
        mel = li.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        mel = torch.from_numpy(mel).float()
        self.register_buffer("mel", mel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mel = self.mel.type_as(x)
        y = torch.einsum("bft,mf->bmt", x, mel)
        return y


class MultiScaleSTFT(nn.Module):

    def __init__(
        self,
        scales: Sequence[int],
        sample_rate: int,
        magnitude: bool = True,
        normalized: bool = False,
        num_mels: int | None = None,
    ) -> None:
        super().__init__()
        self.scales = scales
        self.magnitude = magnitude
        self.num_mels = num_mels

        self.stfts = []
        self.mel_scales = []
        for scale in scales:
            self.stfts.append(
                torchaudio.transforms.Spectrogram(
                    n_fft=scale,
                    win_length=scale,
                    hop_length=scale // 4,
                    normalized=normalized,
                    power=None,
                )
            )
            if num_mels is not None:
                self.mel_scales.append(
                    MelScale(
                        sample_rate=sample_rate,
                        n_fft=scale,
                        n_mels=num_mels,
                    )
                )
            else:
                self.mel_scales.append(None)

        self.stfts = nn.ModuleList(self.stfts)
        self.mel_scales = nn.ModuleList(self.mel_scales)

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x = rearrange(x, "b c t -> (b c) t")
        stfts = []
        for stft, mel in zip(self.stfts, self.mel_scales):
            y = stft(x)
            if mel is not None:
                y = mel(y)
            if self.magnitude:
                y = y.abs()
            else:
                y = torch.stack([y.real, y.imag], -1)
            stfts.append(y)

        return stfts


class AudioDistance(nn.Module):

    def __init__(
        self, multiscale_stft: MultiScaleSTFT, log_epsilon: float = 1e-7
    ) -> None:
        super().__init__()
        self.multiscale_stft = multiscale_stft
        self.log_epsilon = log_epsilon

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        stfts_x = self.multiscale_stft(x)
        stfts_y = self.multiscale_stft(y)
        distance = 0.0

        for x, y in zip(stfts_x, stfts_y):
            logx = torch.log(x + self.log_epsilon)
            logy = torch.log(y + self.log_epsilon)

            lin_distance = mean_difference(x, y, norm="L2", relative=True)
            log_distance = mean_difference(logx, logy, norm="L1")

            distance = distance + lin_distance + log_distance

        return {"spectral_distance": distance}
