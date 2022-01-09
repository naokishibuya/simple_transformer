import torch.nn as nn
from torch import Tensor
from .attention import MultiHeadAttention
from .feed_forward import PositionwiseFeedForward


class Encoder(nn.Module):
    def __init__(self,
                 num_blocks: int,
                 num_heads:  int,
                 dim_embed:  int,
                 dim_pffn:   int,
                 drop_prob:  float) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [EncoderBlock(num_heads, dim_embed, dim_pffn, drop_prob) for _ in range(num_blocks)]
        )
        self.layer_norm = nn.LayerNorm(dim_embed)
        
    def forward(self, x: Tensor, x_mask: Tensor):
        for block in self.blocks:
            x = block(x, x_mask)
        x = self.layer_norm(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self,
                 num_heads: int,
                 dim_embed: int,
                 dim_pwff:  int,
                 drop_prob: float) -> None:
        super().__init__()

        # Self-attention
        self.self_atten = MultiHeadAttention(num_heads, dim_embed, drop_prob)
        self.layer_norm1 = nn.LayerNorm(dim_embed)

        # Point-wise feed-forward
        self.feed_forward = PositionwiseFeedForward(dim_embed, dim_pwff, drop_prob)
        self.layer_norm2 = nn.LayerNorm(dim_embed)

    def forward(self, x: Tensor, x_mask: Tensor) -> Tensor:
        x = x + self.sub_layer1(x, x_mask)
        x = x + self.sub_layer2(x)
        return x

    def sub_layer1(self, x: Tensor, x_mask: Tensor) -> Tensor:
        x = self.layer_norm1(x)
        x = self.self_atten(x, x, x_mask)
        return x
    
    def sub_layer2(self, x: Tensor) -> Tensor:
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        return x
