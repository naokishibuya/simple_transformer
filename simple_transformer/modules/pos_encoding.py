import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """ Encodes positional information into word embeddings.

    Reference: https://naokishibuya.medium.com/positional-encoding-286800cce437
    """
    def __init__(self, max_positions: int, dim_embed: int, drop_prob: float) -> None:
        super().__init__()

        # The number of embedding vector dimensions must be even as
        # we assign a pair of sine and cosine values per every two dimensions.
        assert dim_embed % 2 == 0

        # Pre-compute the positional encodings
        
        # Slow but readable version
        #
        # pe = torch.zeros(max_positions, dim_embed)
        # for pos in range(max_positions):
        #    for i in range(0, dim_embed, 2):
        #        theta = pos / (10000 ** (i / dim_embed))
        #        pe[pos, i    ] = math.sin(theta)
        #        pe[pos, i + 1] = math.cos(theta)

        # Inspired by https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        position = torch.arange(max_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_embed, 2) * (-math.log(10000.0) / dim_embed))

        pe = torch.zeros(max_positions, dim_embed)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)           # Add a dimension for batch_size => shape: (1, max_positions, dim_embed)
        self.register_buffer('pe', pe) # Register as non-learnable parameters
        
        self.dropout = nn.Dropout(p=drop_prob)
        
    def forward(self, x: Tensor) -> Tensor:
        max_sentence_length = x.size(1)          # The max length of sentences in the current batch
        x = x + self.pe[:, :max_sentence_length] # Add positional encoding up to the max sentence length
        x = self.dropout(x)
        return x
