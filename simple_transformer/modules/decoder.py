import torch.nn as nn
from torch import Tensor
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class Decoder(nn.Module):
    def __init__(self,
                 num_blocks: int,
                 num_heads:  int,
                 dim_embed:  int,
                 dim_pffn:   int,
                 drop_prob:  float) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [DecoderBlock(num_heads, dim_embed, dim_pffn, drop_prob) for _ in range(num_blocks)]
        )
        self.layer_norm = nn.LayerNorm(dim_embed)

    def forward(self, x: Tensor, x_mask: Tensor, y: Tensor, y_mask: Tensor) -> Tensor:
        for block in self.blocks:
            y = block(y, y_mask, x, x_mask)
        y = self.layer_norm(y)
        return y


class DecoderBlock(nn.Module):
    def __init__(self,
                 num_heads: int,
                 dim_embed: int,
                 dim_pwff:  int,
                 drop_prob: float) -> None:
        super().__init__()

        # Self-attention
        self.self_attn = MultiHeadAttention(num_heads, dim_embed, drop_prob)
        self.layer_norm1 = nn.LayerNorm(dim_embed)

        # Target-source
        self.tgt_src_attn = MultiHeadAttention(num_heads, dim_embed, drop_prob)
        self.layer_norm2 = nn.LayerNorm(dim_embed)

        # Position-wise
        self.feed_forward = PositionwiseFeedForward(dim_embed, dim_pwff, drop_prob)
        self.layer_norm3 = nn.LayerNorm(dim_embed)

    def forward(self, y, y_mask, x, x_mask) -> Tensor:
        y = y + self.sub_layer1(y, y_mask)
        y = y + self.sub_layer2(y, x, x_mask)
        y = y + self.sub_layer3(y)
        return y

    def sub_layer1(self, y: Tensor, y_mask: Tensor) -> Tensor:
        y = self.layer_norm1(y)
        y = self.self_attn(y, y, y_mask)
        return y

    def sub_layer2(self, y: Tensor, x: Tensor, x_mask: Tensor) -> Tensor:
        y = self.layer_norm2(y)
        y = self.tgt_src_attn(y, x, x_mask)
        return y

    def sub_layer3(self, y: Tensor) -> Tensor:
        y = self.layer_norm3(y)
        y = self.feed_forward(y)
        return y
