import torch


def rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float = 10000.0,
    device=None,
):
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=device,
                dtype=torch.float32,
            )
            / head_dim
        )
    )

    positions = torch.arange(
        seq_len,
        device=device,
        dtype=torch.float32,
    )

    freqs = torch.outer(positions, inv_freq)

    emb = torch.cat([freqs, freqs], dim=-1)

    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2

    return torch.cat(
        [-x[..., half:], x[..., :half]],
        dim=-1,
    )


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    return x * cos + rotate_half(x) * sin