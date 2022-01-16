import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """ Multi-head attention runs multiple attention calculations in parallel.
    """
    def __init__(self, num_heads: int, dim_embed: int, drop_prob: float) -> None:
        super().__init__()
        assert dim_embed % num_heads == 0

        # num_head x dim_head = dim_embed
        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.dim_head = dim_embed // num_heads

        # Linear operations and dropout
        self.query  = nn.Linear(dim_embed, dim_embed)
        self.key    = nn.Linear(dim_embed, dim_embed)
        self.value  = nn.Linear(dim_embed, dim_embed)
        self.output = nn.Linear(dim_embed, dim_embed)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor, y: Tensor, mask: Tensor=None) -> Tensor:
        """ The main idea is to apply multiple attention operations on the same set of inputs.
        Instead of using seperate linear operations and concatenate them later, we do a single
        linear operation per query, key, value and rearange them to be independent heads by
        partitioning dim_embed into num_heads x dim_head.

        Applies the linear operations to extract query, key, and value tensors.
        Then, divide dim_embed into num_heads x dim_head

        """
        # linear transformation in one shot per query, key, value
        query = self.query(x)
        key   = self.key  (y)
        value = self.value(y)

        # Note: max here is within a batch and it's either for target or source batch
        # (batch_size, max_sequence_length, dim_embed) =>
        # (batch_size, max_sequence_length, num_heads, dim_head) =>
        # (batch_size, num_heads, max_sequence_length, dim_head)
        batch_size = x.size(0)
        query = query.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        key   = key  .view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        
        if mask is not None:
            # Mask needs to have an extra dimension to be broadcastable across multiple heads
            # - Encoder self-attention: (batch_size, 1,                       1, max_src_sequence_length)
            # - Decoder self-attention: (batch_size, 1, max_tgt_sequence_length, max_tgt_sequence_length)
            mask = mask.unsqueeze(1)

        # Applies the attention function on all heads in parallel
        attn = attention(query, key, value, mask)

        # Restores the original shapes:
        # (batch_size, num_heads, max_sequence_length, dim_head) =>
        # (batch_size, max_sequence_length, num_heads, dim_head) =>
        # (batch_size, max_sequence_length, dim_embed)
        attn = attn.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_embed)
        
        # Finally, applies one more linear operation and dropout
        out = self.dropout(self.output(attn))
        return out


def attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor=None) -> Tensor:
    """ Attention calculator used by the multi-headed attention.
    
    [1] For self-attention, query (Q), key (K), value (V) all have the same shape:

    - Q, K, V: (batch_size, num_heads, max_sequence_length, dim_head)

    Note: these max sequence length is determined per batch.

    Attention scores will be calculated by the scaled dot-product divided by the square-root of dim_head.

    - Scores : (batch_size, num_heads, max_sequence_length, max_sequence_length)

    It tells us which token is relevant to which tokens within the same sequence.

    [2] For target-source attention, Q and K may have different max sequence length:

    - Q      : (batch_size, num_heads, max_tgt_sequence_length, dim_head)
    - K, V   : (batch_size, num_heads, max_src_sequence_length, dim_head)

    Note: these max sequence lengths are determined per batch.

    - Scores : (batch_size, num_heads, max_tgt_sequence_length, max_src_sequence_length)

    It tells us which token in the target sequence is relevant to which tokens in the source sequence.

    [3] Mask is used to make certain tokens excluded from attention scores.
    
    For Encoder, PAD_IDX tokens are masked which has the following broadcastable shape:

    - Encoder self-attention : (batch_size, 1,                       1, max_src_sequence_length)

      Note:
      - The second dimension has 1 which is broadcasted across the number of heads.
      - The third  dimension has 1 which is broadcasted across the number of source sequence tokens.

    For Decoder, PAD_IDX and subsequent tokens are masked:
   
    - Decoder self-attention : (batch_size, 1, max_tgt_sequence_length, max_tgt_sequence_length)

      Note:
      - The second dimension has 1 which is broadcasted across the number of heads.

    For Decoder-Encoder link, PAD_IDX tokens are masked in the source tokens:

    - Target-source attention: (batch_size, 1,                       1, max_src_sequence_length)

      Note:
      - The second dimension has 1 which is broadcasted across the number of heads.
      - The third  dimension has 1 which is broadcasted across the number of target sequence tokens.
    """
    sqrt_dim_head = query.shape[-1]**0.5 # sqrt(dim_head)

    # Scaled Dot-Product by matrix operation: Q K^T / sqrt(dim_head)
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / sqrt_dim_head
    
    if mask is not None:
        # Sets large negative value to masked token positions - softmax will give effectively zero probability to them.
        scores = scores.masked_fill(mask==0, -1e9)
    
    # Attention weighted value
    weight = F.softmax(scores, dim=-1)    
    return torch.matmul(weight, value)
