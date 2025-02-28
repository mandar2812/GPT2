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
        query_len = x.size(-2)

        # Create distance matrix (relative positions)
        bias = self.get_positional_bias(query_len, device=x.device)

        # Generate slopes for each attention head
        slopes = self.get_slopes(self.num_heads, device=x.device).view(
            self.num_heads, 1, 1
        )

        # Apply slopes to bias values (this creates the ALiBi mask)
        alibi_mask = bias * slopes  # (num_heads, query_len, query_len)

        return alibi_mask  # No need to expand since it's applied per head in attention

    def get_positional_bias(self, query_len: int, device=None):
        arange = torch.arange(query_len, device=device)
        bias = arange.view(1, 1, query_len) - arange.view(
            1, query_len, 1
        )  # (1, query_len, query_len)
        return torch.tril(bias).float()  # Convert to absolute values

    def get_slopes(self, num_heads: int, device=None):
        """
        Get the predefined slopes for ALiBi.
        The slopes are chosen based on powers of 2 to create different attention behaviors per head.
        """
        return (2 ** (-8 / num_heads)) ** torch.arange(1, num_heads + 1, device=device)
