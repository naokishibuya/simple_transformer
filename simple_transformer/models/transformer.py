import torch.nn as nn
from torch import Tensor
from ..modules import Embedding, PositionalEncoding, Encoder, Decoder


class Transformer(nn.Module):
    """ The transformer encoder-decoder architecture implementation.

    Reference: https://naokishibuya.medium.com/transformers-encoder-decoder-434603d19e1
    """
    def __init__(self,
                 input_vocab_size:  int,
                 output_vocab_size: int,
                 max_positions:     int,
                 num_blocks:        int,
                 num_heads:         int,
                 dim_embed:         int,
                 dim_pffn:          int,
                 drop_prob:         float) -> None:
        super().__init__()

        # Input embeddings, positional encoding, and encoder
        self.input_embedding = Embedding(input_vocab_size, dim_embed)
        self.input_pos_encoding = PositionalEncoding(max_positions, dim_embed, drop_prob)
        self.encoder = Encoder(num_blocks, num_heads, dim_embed, dim_pffn, drop_prob)

        # Output embeddings, positional encoding, decoder, and projection to vocab size dimension
        self.output_embedding = Embedding(output_vocab_size, dim_embed)
        self.output_pos_encoding = PositionalEncoding(max_positions, dim_embed, drop_prob)
        self.decoder = Decoder(num_blocks, num_heads, dim_embed, dim_pffn, drop_prob)
        self.projection = nn.Linear(dim_embed, output_vocab_size)

        # Initialize parameters
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x: Tensor, y: Tensor, x_mask: Tensor=None, y_mask: Tensor=None) -> Tensor:
        x = self.encode(x, x_mask)
        y = self.decode(x, y, x_mask, y_mask)
        return y

    def encode(self, x: Tensor, x_mask: Tensor=None) -> Tensor:
        x = self.input_embedding(x)
        x = self.input_pos_encoding(x)
        x = self.encoder(x, x_mask)
        return x

    def decode(self, x: Tensor, y: Tensor, x_mask: Tensor=None, y_mask: Tensor=None) -> Tensor:
        y = self.output_embedding(y)
        y = self.output_pos_encoding(y)
        y = self.decoder(x, x_mask, y, y_mask)
        return self.projection(y)
