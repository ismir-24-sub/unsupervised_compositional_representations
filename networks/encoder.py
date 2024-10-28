from einops import rearrange
from torch import nn
import torch
import gin
from networks.unet import EncoderBlock, ResConvBlock
from einops.layers.torch import Rearrange
from networks.blocks import get_activation, Conv1d

@gin.configurable
class EncoderAudio(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        channels: list[int],
        kernel: list[int] | int,
        strides: list[int] | int,
        attn: list[int] | int,
        activation: str,
        norm: str,
        final_fn: str,
        add_gru: bool,
        bias: bool,
        p_drop: float,
    ):
        super().__init__()
        if isinstance(strides, int):
            strides = [strides] * len(channels)
        assert len(channels) == len(
            strides
        ), "channels and strides must have the same length"
        if isinstance(attn, int):
            attn = [attn] * len(channels)
        assert len(channels) == len(attn), "channels and attn must have the same length"
        if isinstance(kernel, int):
            kernel = [kernel] * len(channels)
        assert len(channels) == len(
            kernel
        ), "channels and kernel must have the same length"
        channels = [in_dim] + channels
        self.blocks = nn.ModuleList()
        for in_c, out_c, stride, use_attn, ker in zip(
            channels[:-1],
            channels[1:],
            strides,
            attn,
            kernel,
        ):
            self.blocks.append(
                EncoderBlock(
                    in_c=in_c,
                    out_c=out_c,
                    cond_c=0,
                    tcond_c=0,
                    kernel=ker,
                    stride=stride,
                    activation=activation,
                    norm=norm,
                    use_attn=use_attn,
                    p_drop=p_drop,
                    bias=bias,
                )
            )
        self.final_block = ResConvBlock(
            c_in=channels[-1],
            c_mid=channels[-1],
            c_out=channels[-1],
            tc_cond=0,
            c_cond=0,
            kernel_size=3,
            activation=activation,
            norm=norm,
            bias=bias,
            p_drop=0.0,
        )
        self.use_gru = add_gru
        self.r1 = Rearrange("b c t -> b t c")
        self.r2 = Rearrange("b t c -> b c t")
        self.gru = (
            nn.GRU(channels[-1], channels[-1], batch_first=True, num_layers=1)
            if add_gru
            else nn.Identity()
        )
        self.out_fn = get_activation(final_fn, channels[-1])

    def forward(self, x: torch.Tensor):
        for block in self.blocks:
            x, _ = block(x=x, t=None, time_cond=None)
        x = self.final_block(x=x, cond=None, time_cond=None)
        if self.use_gru:
            x = self.r1(x)
            x, _ = self.gru(x)
            x = self.r2(x)
        return self.out_fn(x)


@gin.configurable
class MultiEncoder(nn.Module):
    def __init__(
        self,
        *,
        n_comp: int,
        in_dim: int,
        channels: list[int],
        kernel: int,
        strides: list[int],
        attn: list[bool],
        activation: str,
        norm: str,
        out_fn: str,
        add_gru: bool,
        bias: bool,
        p_drop: float,
    ):
        super().__init__()
        self.n_comp = n_comp
        self.encoders = nn.ModuleList()
        for _ in range(n_comp):
            self.encoders.append(
                EncoderAudio(
                    in_dim=in_dim,
                    channels=channels,
                    kernel=kernel,
                    strides=strides,
                    attn=attn,
                    activation=activation,
                    norm=norm,
                    final_fn="id",
                    add_gru=add_gru,
                    bias=bias,
                    p_drop=p_drop,
                )
            )
        self.latent_dim = channels[-1]
        self.out_fn = get_activation(out_fn, channels[-1])

    def forward(self, x: torch.Tensor):
        res = []
        b, *_ = x.shape
        for i, encoder in enumerate(self.encoders):
            out = encoder(x)
            res.append(out)
        res = torch.stack(res, dim=1)
        assert res.ndim == 4, f"Found {res.ndim} dimensions, expected 4"
        _, nc, _, _ = res.shape
        assert nc == self.n_comp
        return self.out_fn(res)  # b, nc, c, t

    def reshape_components(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b (nc c) t -> b nc c t", nc=self.n_comp, c=self.latent_dim)

    def merge_components(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b nc c t -> b (nc c) t")


@gin.configurable
class CommonBlockEncoder(nn.Module):
    def __init__(
        self,
        *,
        n_comp: int,
        in_dim: int,
        channels: list[int],
        kernel: list[int] | int,
        strides: list[int],
        attn: list[bool],
        activation: str,
        norm: str,
        out_fn: str,
        add_gru: bool,
        bias: bool,
        p_drop: float,
    ):
        super().__init__()
        self.n_comp = n_comp
        self.encoder = EncoderAudio(
            in_dim=in_dim,
            channels=channels,
            kernel=kernel,
            strides=strides,
            activation=activation,
            norm=norm,
            attn=attn,
            final_fn="id",
            add_gru=add_gru,
            bias=bias,
            p_drop=p_drop,
        )
        self.final = nn.ModuleList(
            [
                ResConvBlock(
                    c_in=channels[-1],
                    c_mid=channels[-1],
                    c_out=channels[-1],
                    c_cond=0,
                    tc_cond=0,
                    kernel_size=kernel,
                    activation=activation,
                    norm=norm,
                    bias=bias,
                    p_drop=p_drop,
                ),
                ResConvBlock(
                    c_in=channels[-1],
                    c_mid=channels[-1],
                    c_out=channels[-1],
                    c_cond=0,
                    tc_cond=0,
                    kernel_size=kernel,
                    activation=activation,
                    norm=norm,
                    bias=bias,
                    p_drop=0.0,
                ),
            ]
        )
        self.final_conv = Conv1d(channels[-1], channels[-1], 3, padding="same")
        self.out_fn = get_activation(out_fn, channels[-1])
        assert (
            channels[-1] % n_comp == 0
        ), "The number of channels in the last layer must be a multiple of n_comp"
        self.latent_dim = channels[-1] // n_comp

    def forward(self, x: torch.Tensor):
        res = self.encoder(x)
        for l in self.final:
            res = l(x=res, cond=None, time_cond=None)
        res = self.final_conv(res)
        return rearrange(
            self.out_fn(res),
            "b (n c) t -> b n c t",
            n=self.n_comp,
            c=self.latent_dim,
        )

    def reshape_components(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b (nc c) t -> b nc c t", nc=self.n_comp, c=self.latent_dim)

    def merge_components(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b nc c t -> b (nc c) t")
    
if __name__ == "__main__":

    com = MultiEncoder(
        n_comp=2,
        in_dim=128,
        channels=[256, 512, 1],
        kernel=3,
        strides=1,
        attn=0,
        activation="silu",
        norm="none",
        out_fn="id",
        add_gru=False,
        bias=True,
        p_drop=0.0,
    )

    n_params = sum(p.numel() for p in com.parameters()) / 1e6
    print(f"Number of parameters: {n_params:.2f}M")

    x = torch.randn(1, 128, 512)
    y = com(x)
    print(y.shape)