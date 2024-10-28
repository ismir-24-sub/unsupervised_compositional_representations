# some snippets adapted / taken from https://github.com/Stability-AI/stable-audio-tools/blob/main/stable_audio_tools/models/blocks.py

from einops import rearrange
import torch
from torch import nn
import gin

@gin.configurable
def weight_norm(module: nn.Module, norm: str = "id"):
    if norm == "id":
        return module
    elif norm == "wn":
        return nn.utils.parametrizations.weight_norm(module)
    else:
        raise ValueError(f"Invalid norm: {norm}")


# @persistent_class
def Conv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


# @persistent_class
def ConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


# @persistent_class
def Linear(*args, **kwargs):
    return weight_norm(nn.Linear(*args, **kwargs))

def Downsample1d_2(
    in_channels: int, out_channels: int, factor: int, kernel_multiplier: int = 2
) -> nn.Module:
    assert kernel_multiplier % 2 == 0, "Kernel multiplier must be even"

    return Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=factor * kernel_multiplier + 1,
        stride=factor,
        padding=factor * (kernel_multiplier // 2),
    )


def Upsample1d_2(
    in_channels: int,
    out_channels: int,
    factor: int,
    use_nearest: bool = False,
    bias: bool = True,
) -> nn.Module:

    if factor == 1:
        return Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

    if use_nearest:
        return nn.Sequential(
            nn.Upsample(scale_factor=factor, mode="nearest-exact"),
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=bias,
            ),
        )
    else:
        return ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=factor * 2,
            stride=factor,
            padding=factor // 2 + factor % 2,
            output_padding=factor % 2,
            bias=bias,
        )

class SelfAttention1d(nn.Module):
    def __init__(self, c_in: int, n_head: int = 1, dropout_rate: float = 0.0):
        super().__init__()
        assert c_in % n_head == 0
        self.norm = nn.GroupNorm(1, c_in)
        self.n_head = n_head
        self.qkv_proj = Conv1d(c_in, c_in * 3, 1)
        self.out_proj = Conv1d(c_in, c_in, 1)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

    def forward(self, inpt: torch.Tensor):
        n, c, s = inpt.shape
        # [n, c * 3, s]
        qkv = self.qkv_proj(self.norm(inpt))
        # [n, 3 * h, s, c // h]
        qkv = qkv.view([n, self.n_head * 3, c // self.n_head, s]).transpose(2, 3)
        # each [n, h, s, c // h]
        q, k, v = qkv.chunk(3, dim=1)
        scale = k.shape[3] ** -0.25
        att = ((q * scale) @ (k.transpose(2, 3) * scale)).softmax(3)
        y = (att @ v).transpose(2, 3).contiguous().view([n, c, s])
        return inpt + self.dropout(self.out_proj(y))
    
@gin.configurable(allowlist=["groups"])
def get_norm_layer(norm: str, channels: int, groups: int = 16):
    assert norm in ["bn", "gn", "ln", "none"]
    if norm == "bn":
        return nn.BatchNorm1d(channels)
    elif norm == "gn":
        n_groups = min(groups, channels)
        if channels % n_groups != 0:
            print(f"warning. {channels} not divisible by {n_groups} - not normalizing")
            return nn.Identity()
        return nn.GroupNorm(min(groups, channels), channels)
    elif norm == "ln":
        return nn.LayerNorm(channels)
    else:
        return nn.Identity()

def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * pow(torch.sin(x * alpha), 2)

# Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/activations.py under MIT license
# License available in LICENSES/LICENSE_NVIDIA.txt
class SnakeBeta(nn.Module):

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True
    ):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)
        return x


def get_activation(activation: str, channels: int):
    assert activation in ["relu", "silu", "leaky", "snake", "tanh", "id"]
    if activation == "relu":
        return nn.ReLU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "leaky":
        return nn.LeakyReLU(0.2)
    elif activation == "snake":
        return SnakeBeta(channels)
    elif activation == "tanh":
        return nn.Tanh()
    elif "id" in activation:
        return nn.Identity()


class FiLM(nn.Module):
    def __init__(self, c_cond: int, c_out: int, act: str):
        super().__init__()
        self.linear = nn.Sequential(
            Linear(c_cond, c_out, bias=False),
            get_activation(act, c_out),
            Linear(c_out, c_out * 2, bias=False),
        )
        self.act = get_activation(act, c_out)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        gamma, beta = self.linear(cond).chunk(2, dim=-1)
        gamma = rearrange(gamma, "b c -> b c 1")
        beta = rearrange(beta, "b c -> b c 1")
        x = x * gamma + beta
        return self.act(x)