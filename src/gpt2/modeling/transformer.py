import torch
import torch.nn as nn
import torch.utils.checkpoint
from functools import partial
from src.gpt2.utils.fusing import LayerNorm
from src.gpt2.modeling import (
    PadMasking,
    FutureMasking,
    AttentionLayer,
    Past,
    TokenEmbedding,
    PositionwiseFeedForward,
)
from typing import Optional, Tuple, List, Union


class TransformerLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               float           (..., seq_len, dims)
    past (*)        float           (..., past_len, dims)
    mask            bool            (..., seq_len, past_len + seq_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (*)    float           (..., past_len + seq_len, dims)
    ===========================================================================
    """

    def __init__(
        self,
        heads: int,
        dims: int,
        rate: int,
        dropout: float = 0.1,
        scaled_softmax: bool = False,
    ):
        super().__init__()
        self.attn = AttentionLayer(heads, dims, dropout, scaled_softmax)
        self.ff = PositionwiseFeedForward(dims, rate, dropout)
        self.ln_attn = LayerNorm(dims)
        self.ln_ff = LayerNorm(dims)

    def forward(
        self,
        x: torch.Tensor,
        past: Optional[Past] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Past]]:
        # Layer normalizations are performed before the layers respectively.
        a = self.ln_attn(x)
        a, past = self.attn(a, a, a, past, mask)

        x = x + a
        x = x + self.ff(self.ln_ff(x))

        return x if self.training else (x, past)


class Transformer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               long            (..., seq_len)
    past (**)       float           (..., past_len, dims)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (**)   float           (..., past_len + seq_len, dims)
    ===========================================================================
    """

    def __init__(
        self,
        layers: int,
        pad_idx: int,
        words: int,
        heads: int,
        dims: int,
        rate: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
        scaled_softmax: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.pad_masking = PadMasking(pad_idx)
        self.future_masking = FutureMasking()

        self.token_embedding = TokenEmbedding(words, dims, padding_idx=pad_idx)
        self.dropout_embedding = nn.Dropout(dropout)

        self.transformers = nn.ModuleList(
            [
                TransformerLayer(heads, dims, rate, dropout, scaled_softmax)
                for _ in range(layers)
            ]
        )
        self.ln_head = LayerNorm(dims)

    def forward(
        self,
        x: torch.Tensor,
        past: Optional[List[Past]] = None,
        use_grad_ckpt: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Past]]]:
        offset = past[0][0].size(-2) if past is not None else 0

        # Create masking tensor.
        mask = self.pad_masking(x, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(x, offset)

        # Use token embedding.
        x = self.token_embedding(x)  # + self.positional_embedding(x, offset)
        x = self.dropout_embedding(x)

        # Apply transformer layers sequentially.
        present = []
        for i, transformer in enumerate(self.transformers):
            if self.training and use_grad_ckpt:
                transformer = partial(torch.utils.checkpoint.checkpoint, transformer)

            x = transformer(x, past[i] if past is not None else None, mask)

            if not self.training:
                present.append(x[1])
                x = x[0]

        x = self.ln_head(x)
        x = self.token_embedding(x, transposed=True)

        return x if self.training else (x, present)
