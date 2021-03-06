import torch
import torch.nn as nn
from torch import Tensor
from ..data import PAD_IDX


class TranslationLoss(nn.Module):
    """ A wrapper for `nn.CrossEntropyLoss`.

    The loss will be averaged over non-PAD_IDX target tokens.

    Optionally, it applies the label smoothing.        
    """
    def __init__(self, label_smoothing: float=0.0) -> None:
        super().__init__()
        self.loss_func = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=label_smoothing)

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """ CrossEntropyLoss expects for each token position to have unnormalized scores for each class.
        In other words, it expects the input shape to be (num_output_tokens, output_vocab_size).

        So, logits (batch_size, max_target_sequence_length, output_vocab_size) must be reshaped
        into (batch_size * max_target_sequence_length, output_vocab_size).

        As for labels, CrossEntropyLoss expects outputs to be a list of target indicies.
        So, targets (batch_size, max_target_sequence_length) must be reshaped
        into (batch_size * max_target_sequence_length).
        """
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size) # (batch_size * max_sequence_length, output_vocab_size)
        target = target.reshape(-1).long()      # (batch_size * max_sequence_length)

        return self.loss_func(logits, target)
