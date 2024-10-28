import encodec
from torch import nn
import torch
import numpy as np
import music2latent as m2l
import gin

@gin.configurable
class Encodec(nn.Module):

    def __init__(
        self, sr: int = 24000, bandwidth: float = 24.0
    ):
        super().__init__()
        self.sr = sr
        assert sr in [24000, 48000]
        if sr == 24000:
            self.model = encodec.model.EncodecModel.encodec_model_24khz()
        else:
            self.model = encodec.model.EncodecModel.encodec_model_48khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.eval()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        codes, _ = self.model._encode_frame(x)
        codes = codes.transpose(0, 1)
        z = self.model.quantizer.decode(codes)
        return z

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decoder(z)


@gin.configurable
class SonyAE(nn.Module):

    def __init__(self, sr: int = 44100, fp16: bool = True, device: str = "cpu"):
        super().__init__()
        self.sr = sr
        self.fp16 = fp16
        self.device = device
        self.model = m2l.EncoderDecoder(device=device)

    def encode(self, x: torch.Tensor):
        return self.model.encode(x)

    def decode(self, z: torch.Tensor):
        if self.fp16:
            z = z.half()
        return self.model.decode(z.to(self.device))
