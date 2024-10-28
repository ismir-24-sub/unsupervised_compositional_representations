import gin
import gin.config
import torch
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional
from networks.blocks import Conv1d, Linear, Downsample1d_2, Upsample1d_2, get_norm_layer, get_activation, SelfAttention1d, FiLM

HEAD_DIM = 32  # for attention

class TimeEmbedding(nn.Module):
    def __init__(
        self,
        num_channels: int,
        max_positions: int = 10_000,
        endpoint: bool = False,
        factor: float = 1.0,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            PositionalEmbedding(num_channels, max_positions, endpoint, factor),
            Linear(num_channels, 2 * num_channels),
            nn.SiLU(),
            Linear(2 * num_channels, num_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        num_channels: int,
        max_positions: int,
        factor: float,
        endpoint: bool = False,
        rearrange: bool = False,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint
        self.factor = factor
        self.rearrange = (
            Rearrange("b (f c) -> b (c f)", f=2) if rearrange else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        x = x * self.factor
        freqs = torch.arange(
            start=0,
            end=self.num_channels // 2,
            device=x.device,
        ).float()
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return self.rearrange(x)


def interpolate_t_cond(*, x: torch.Tensor, time_cond: torch.Tensor | None):
    *_, t_time = time_cond.shape
    *_, x_time = x.shape
    if t_time == x_time:
        return time_cond
    factor = x_time / t_time
    return functional.interpolate(time_cond, scale_factor=factor, mode="nearest-exact")


class ResConvBlock(nn.Module):
    def __init__(
        self,
        *,
        c_in: int,
        c_mid: int,
        c_out: int,
        c_cond: int,
        tc_cond: int,
        kernel_size: int,
        norm: str,
        activation: str,
        p_drop: float,
        bias: bool,
    ):
        super().__init__()
        self.skip = (
            nn.Identity()
            if c_in == c_out
            else Conv1d(c_in, c_out, 1, padding="same", bias=bias)
        )
        self.block1 = nn.Sequential(
            Conv1d(c_in, c_mid, kernel_size, padding="same", bias=bias),
            get_norm_layer(norm, c_mid),
            get_activation(activation, c_mid),
        )
        self.film = FiLM(c_cond, c_mid, activation) if c_cond != 0 else nn.Identity()
        if tc_cond != 0:
            self.tblock = nn.Sequential(
                Conv1d(
                    tc_cond,
                    tc_cond,
                    kernel_size=kernel_size,
                    padding="same",
                    bias=bias,
                ),
                get_activation(activation, tc_cond),
            )
        else:
            self.tblock = nn.Identity()
        self.drop = nn.Dropout(p_drop)
        self.block2 = nn.Sequential(
            Conv1d(c_mid + tc_cond, c_out, kernel_size, padding="same", bias=bias),
            get_norm_layer(norm, c_out),
            get_activation(activation, c_out),
            Conv1d(c_out, c_out, 1, padding="same", bias=bias),
        )

    def block(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None,
        time_cond: torch.Tensor | None,
    ):
        x = self.block1(x)
        if cond is not None:
            x = self.film(x, cond)
        if time_cond is not None:
            time_cond = interpolate_t_cond(x=x, time_cond=time_cond)
            time_cond = self.tblock(time_cond)
            x = torch.cat([x, time_cond], dim=1)
        x = self.drop(x)
        return self.block2(x)

    def forward(
        self,
        *,
        x: torch.Tensor,
        cond: torch.Tensor | None,
        time_cond: torch.Tensor | None,
    ):
        return self.skip(x) + self.block(x, cond, time_cond)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        in_c: int,
        out_c: int,
        cond_c: int,
        tcond_c: int,
        kernel: int,
        stride: int,
        norm: str,
        activation: str,
        use_attn: bool,
        p_drop: float,
        bias: bool,
    ):
        super().__init__()
        self.conv = ResConvBlock(
            c_in=in_c,
            c_mid=in_c,
            c_out=in_c,
            c_cond=cond_c,
            tc_cond=tcond_c,
            kernel_size=kernel,
            norm=norm,
            activation=activation,
            p_drop=p_drop,
            bias=bias,
        )
        self.attn = (
            SelfAttention1d(in_c, in_c // HEAD_DIM) if use_attn else nn.Identity()
        )
        self.down = (
            Downsample1d_2(in_c, out_c, stride)
            if stride > 1
            else Conv1d(in_c, out_c, 3, padding="same", bias=bias)
        )

    def forward(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor | None,
        time_cond: torch.Tensor | None,
    ):
        x = self.conv(x=x, cond=t, time_cond=time_cond)
        x = self.attn(x)
        return self.down(x), x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        *,
        in_c: int,
        out_c: int,
        cond_c: int,
        tcond_c: int,
        skip_c: int,
        kernel: int,
        stride: int,
        norm: str,
        activation: str,
        use_attn: bool,
        p_drop: float,
        skip_type: str,
        bias: bool,
        use_nearest: bool,
    ):
        super().__init__()
        assert skip_type in ["cat", "add", "none"]
        self.skip_type = skip_type
        self.up = (
            Upsample1d_2(in_c, out_c, stride, use_nearest=use_nearest, bias=bias)
            if stride > 1
            else Conv1d(in_c, out_c, 3, padding="same", bias=bias)
        )
        _skip_c = skip_c if skip_type == "cat" else 0
        self.conv = ResConvBlock(
            c_in=out_c + _skip_c,
            c_mid=out_c,
            c_out=out_c,
            c_cond=cond_c,
            tc_cond=tcond_c,
            kernel_size=kernel,
            norm=norm,
            activation=activation,
            p_drop=p_drop,
            bias=bias,
        )
        self.attn = (
            SelfAttention1d(out_c, out_c // HEAD_DIM) if use_attn else nn.Identity()
        )

    def forward(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        skip: torch.Tensor,
        time_cond: torch.Tensor | None,
    ):
        x = self.up(x)
        if self.skip_type == "cat":
            x = torch.cat([x, skip], dim=1)
        elif self.skip_type == "add":
            x = x + skip
        else:
            pass
        x = self.conv(x=x, cond=t, time_cond=time_cond)
        x = self.attn(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(
        self,
        *,
        in_c: int,
        out_c: int,
        cond_c: int,
        tcond_c: int,
        kernel: int,
        norm: str,
        activation: str,
        use_attn: bool,
        p_drop: float,
        bias: bool,
    ):
        super().__init__()
        self.conv = ResConvBlock(
            c_in=in_c,
            c_mid=out_c,
            c_out=out_c,
            c_cond=cond_c,
            tc_cond=tcond_c,
            kernel_size=kernel,
            norm=norm,
            activation=activation,
            p_drop=p_drop,
            bias=bias,
        )
        self.attn = (
            SelfAttention1d(out_c, out_c // HEAD_DIM) if use_attn else nn.Identity()
        )

    def forward(self, *, x: torch.Tensor, t: torch.Tensor, time_cond: torch.Tensor):
        x = self.conv(x=x, cond=t, time_cond=time_cond)
        x = self.attn(x)
        return x


@gin.configurable
class UNet(nn.Module):
    def __init__(
        self,
        channels: list[int],
        kernels: list[int],
        strides: list[int],
        attn: list[bool] | bool | int | list[int],
        cond_dim: int,
        tcond_dim: int,
        factor: float,
        norm: str,
        activation: str,
        skip_type: str,
        p_drop: float,
        bias: bool,
        use_nearest: bool,
    ):
        super().__init__()
        assert skip_type in ["cat", "add", "none"]
        self.data_dim = channels[0]
        self.cond_dim = cond_dim
        self.factor = factor
        self.noise_emb = TimeEmbedding(cond_dim, factor=factor)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        if isinstance(attn, (bool, int)):
            attn = [attn] * len(kernels)
        for i in range(len(channels) - 1):
            self.encoders.append(
                EncoderBlock(
                    in_c=channels[i],
                    out_c=channels[i + 1],
                    cond_c=cond_dim,
                    tcond_c=tcond_dim,
                    kernel=kernels[i],
                    stride=strides[i],
                    norm=norm,
                    activation=activation,
                    use_attn=attn[i],
                    p_drop=p_drop,
                    bias=bias,
                )
            )
        self.bottleneck = BottleneckBlock(
            in_c=channels[-1],
            out_c=channels[-1],
            cond_c=cond_dim,
            tcond_c=tcond_dim,
            kernel=min(kernels),
            norm=norm,
            activation=activation,
            use_attn=any(attn),
            p_drop=p_drop,
            bias=bias,
        )
        for i in range(len(channels) - 1, 0, -1):
            self.decoders.append(
                DecoderBlock(
                    in_c=channels[i],
                    out_c=channels[i - 1],
                    cond_c=cond_dim,
                    tcond_c=tcond_dim,
                    skip_c=channels[i - 1],
                    kernel=kernels[i - 1],
                    stride=strides[i - 1],
                    norm=norm,
                    activation=activation,
                    use_attn=attn[i - 1],
                    skip_type=skip_type,
                    p_drop=p_drop,
                    bias=bias,
                    use_nearest=use_nearest,
                )
            )

        self.out = nn.Conv1d(self.data_dim, self.data_dim, 3, padding="same", bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ):
        skips = []
        t = self.noise_emb(t)
        for encoder in self.encoders:
            x, skip = encoder(x=x, t=t, time_cond=cond)
            skips.append(skip)
        x = self.bottleneck(x=x, t=t, time_cond=cond)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x=x, t=t, skip=skip, time_cond=cond)
        return self.out(x)


@gin.configurable
class DecompUNet(UNet):
    def __init__(self, n_comp: int, comp_function: str, **kwargs):
        super().__init__(**kwargs)
        self.n_comp = n_comp
        self.comp_function = comp_function

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        component_idx: int | None = None,
    ):
        if component_idx is None:
            cond = rearrange(cond, "b n c t -> (b n) c t")
            x = repeat(x, "b c t -> (b n) c t", n=self.n_comp)
            t = repeat(t, "b -> (b n) ", n=self.n_comp)
        else:
            cond = cond[:, component_idx]
        out = super().forward(x, t, cond.contiguous())
        if component_idx is None:
            components = rearrange(out, "(b n) c t -> b n c t", n=self.n_comp)
            out = reduce(components, "b n c t -> b c t", self.comp_function)
        else:
            components = None
        return dict(out=out, components=components)
    
if __name__ == "__main__":

    comp, c, l = 2, 128, 512

    decomp = DecompUNet(
        n_comp=comp,
        comp_function="mean",
        channels = [c, 128, 256, 256, 512],
        kernels = [5, 3, 3, 3],
        strides = [2, 2, 2, 2],
        attn = [False, False, False, True],
        cond_dim = 128,
        tcond_dim = 1,
        factor = 1000.0,
        norm = "gn",
        activation = "silu",
        skip_type = "cat",
        p_drop = 0.1,
        bias = True,
        use_nearest = False,
    )

    n_params = sum(p.numel() for p in decomp.parameters()) / 1e6
    print(f"Number of parameters: {n_params:.2f}M")

    x = torch.randn(1, c, l)
    c = torch.randn(1, comp, 1, l)
    t = torch.rand(1)

    out = decomp(x, t, c)
    print(out["out"].shape)
    print(out["components"].shape)