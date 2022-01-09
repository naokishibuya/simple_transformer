import torch.nn as nn
from torch import Tensor


class PositionwiseFeedForward(nn.Module):
    """ Applies ReLU non-liniearity in between two linear operations.

    We don't flatten the input x so the linear operation applies on each token position independently and identically.    

    The first linear operation expands the dimensions so that ReLU will not lose too much information.
    The second linear operation restores the original dimensions.

    (batch_size, max_sentence_length, dim_embed) =>
    (batch_size, max_sentence_length, dim_pfnn ) =>
    (batch_size, max_sentence_length, dim_embed)

    Note: max_sentence_length is from the current batch.
    """
    def __init__(self, dim_embed: int, dim_pffn: int, drop_prob: float) -> None:
        super().__init__()
        self.pffn = nn.Sequential(
            nn.Linear(dim_embed, dim_pffn),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),           # See dense_relu_dense in https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py
            nn.Linear(dim_pffn, dim_embed),
            nn.Dropout(drop_prob),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pffn(x)
