from functools import partial
from typing import Callable
from einops import rearrange, reduce
import torch
import gin
from tqdm.auto import tqdm

def standardize(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8
):
    """
    x has shape (b, c, t) and mean and std have shape (c,)
    """
    return (x - mean.to(x)) / (std.to(x) + eps)


def destandardize(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8
):
    """
    x has shape (b, c, t) and mean and std have shape (c,)
    """
    return x * (std.to(x) + eps) + mean.to(x)


def min_max_normalize(x: torch.Tensor, minval: float, maxval: float):
    x = (x - minval) / (maxval - minval)
    return 2 * x - 1


def min_max_denormalize(x: torch.Tensor, minval: float, maxval: float):
    x = 0.5 * x + 0.5
    return x * (maxval - minval) + minval

def set_normalization(norm_dict: dict[str, float | torch.Tensor] | None):
    if norm_dict is not None:
        # if there are 2 keys in the dictionary
        if len(norm_dict) == 2:
            if "mean" in norm_dict and "std" in norm_dict:
                mean = norm_dict["mean"]
                std = norm_dict["std"]
                mean, std = mean[None, :, None], std[None, :, None]
                norm_fn = partial(standardize, mean=mean, std=std)
                denorm_fn = partial(destandardize, mean=mean, std=std)
                print(
                    f"Standardizing data with mean {mean.mean()} and std {std.mean()}"
                )
            else:
                assert "min" in norm_dict and "max" in norm_dict
                min_val = norm_dict["min"]
                max_val = norm_dict["max"]
                norm_fn = partial(min_max_normalize, minval=min_val, maxval=max_val)
                denorm_fn = partial(min_max_denormalize, minval=min_val, maxval=max_val)
                print(f"Normalizing data with min {min_val} and max {max_val}")
        else:
            raise ValueError("Normalization dictionary must have 2 keys")
    else:
        norm_fn = lambda x: x
        denorm_fn = lambda x: x
    return norm_fn, denorm_fn


@gin.configurable(denylist=["get_x", "dl_list"])
def compute_stats(
    get_x: Callable[..., torch.Tensor],
    dl_list: list[torch.utils.data.DataLoader],
    keep: list[str],
    n_batch: int | None,
    save_to: str | None = None,
    reduction_dims: tuple[int] = (0, 2),
) -> dict:
    assert all(
        [k in ["min", "max", "mean", "std"] for k in keep]
    ), "Invalid key in keep"
    
    elems = []
    for dl_idx, dl in enumerate(dl_list):
        for batch_idx, batch in enumerate(
            (
                pbar := tqdm(
                    dl,
                    desc=f"Computing stats: dataset {dl_idx + 1} / {len(dl_list)}",
                    leave=False,
                )
            )
        ):
            x = get_x(batch)
            elems.append(x)
            if n_batch is not None and batch_idx >= n_batch:
                break

    elems = torch.cat(elems, dim=0)
    stats = {}
    if "min" in keep:
        stats["min"] = elems.min()
    if "max" in keep:
        stats["max"] = elems.max()
    if "mean" in keep:
        stats["mean"] = elems.mean(dim=reduction_dims)
    if "std" in keep:
        stats["std"] = elems.std(dim=reduction_dims)

    if save_to is not None:
        torch.save(stats, save_to)
    return stats



def embedding_l2_distances(latents: torch.Tensor):
    """
    Computes the distances between all pairs of elements
    for a tensor of shape (b, n, c).
    """
    if latents.ndim == 4:
        latents = rearrange(latents, "b n c t -> b n (c t)")
    elif latents.ndim == 5:
        latents = rearrange(latents, "b n c h w -> b n (c h w)")
    else:
        assert latents.ndim == 3, f"Invalid number of dimensions {latents.ndim}"
    _, n, _ = latents.shape
    latents = rearrange(latents, "b n c -> b n 1 c")
    perm_latents = rearrange(latents, "b n 1 c -> b 1 n c")
    squared = (latents - perm_latents) ** 2
    sum_squared = reduce(squared, "b n m c -> b n m", "sum", n=n, m=n)
    tril = torch.tril(sum_squared, diagonal=-1)
    sum_tril = reduce(tril, "b n m -> b", "sum", n=n, m=n)
    return sum_tril.mean()