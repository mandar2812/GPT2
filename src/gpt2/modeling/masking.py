import torch
import torch.nn as nn


class PadMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, seq_len + offset)
    ===========================================================================
    """

    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        is_pad = (x == self.pad_idx).unsqueeze(-2)
        shifted = torch.zeros(
            x.size()[:-1]
            + (
                1,
                offset,
            ),
            dtype=torch.bool,
            device=x.device,
        )

        mask = torch.cat((shifted, is_pad), dim=-1)
        return mask.expand(x.shape + mask.shape[-1:])


class FutureMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, seq_len + offset)
    ===========================================================================
    """

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.size(-1)

        # Create shifted upper triangular matrix.
        future = torch.ones(
            (seq_len, seq_len + offset), dtype=torch.bool, device=x.device
        )
        future = future.triu(offset + 1)

        mask = future.view((1,) * (x.ndim - 1) + future.size())
        return mask.expand(x.shape + mask.shape[-1:])

class ALiBiMasking(nn.Module):
    """
    ALiBi (Attention with Linear Biases) Masking
    - Applies a bias that grows linearly with distance instead of using a traditional binary mask.

    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., query_len, heads)
    ---------------------------------------------------------------------------
    output          float           (..., query_len, query_len)
    ===========================================================================
    """

    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(-2)

        # Create distance matrix (relative positions)
        arange = torch.arange(seq_len, device=x.device)
        bias = arange.view(1, 1, seq_len) - arange.view(1, seq_len, 1)  # (1, seq_len, seq_len)
        bias = bias.abs().float()  # Convert to absolute values

        # Generate slopes for each attention head
        slopes = self._get_slopes(self.num_heads).to(x.device).view(self.num_heads, 1, 1)

        # Apply slopes to bias values (this creates the ALiBi mask)
        alibi_mask = bias * slopes  # (num_heads, seq_len, seq_len)

        return alibi_mask  # No need to expand since it's applied per head in attention

    def _get_slopes(self, num_heads: int):
        """
        Get the predefined slopes for ALiBi.
        The slopes are chosen based on powers of 2 to create different attention behaviors per head.
        """
        def get_slopes(n):
            start = 2 ** (-(2 ** -(torch.arange(n) / n)))
            return start

        if num_heads <= 16:
            return get_slopes(num_heads)
        else:
            # If num_heads > 16, split into groups (following the original paper)
            closest_power_of_2 = 2 ** ((num_heads - 1).bit_length() - 1)
            return torch.cat([get_slopes(closest_power_of_2), get_slopes(num_heads - closest_power_of_2)])
