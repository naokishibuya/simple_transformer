import math
import torch.nn as nn
from torch import Tensor


class Embedding(nn.Module):
    """ Word embedding lookup with learnable parameters.

    Reference: https://naokishibuya.medium.com/word-embedding-lookup-826af604dd11
    """
    def __init__(self, vocab_size: int, dim_embed: int) -> None:
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, dim_embed)        
        self.sqrt_dim_embed = math.sqrt(dim_embed)

    def forward(self, x: Tensor) -> Tensor:
        """ Looks up and returns an embedding vector for each token index.

        Args:
            x: A list of tokens (indices).  The maximum sequence length in the batch is the second dimension.

        Shape:
            - x: (batch_size, max_sequence_length)
        """        
        x = self.embedding(x.long()) # (batch_size, max_sequence_length, dim_embed)    
        x = x * self.sqrt_dim_embed  # Scaling
        return x
