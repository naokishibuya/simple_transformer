import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from simple_transformer.optim import TranslationLoss
from simple_transformer.data import PAD_IDX, EOS_IDX


def loss_calc(logits: Tensor, labels: Tensor, label_smoothing: float=0.0) -> float:
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)
    labels = labels.reshape(-1)

    # Negative log-likelihood loss ignoring PAD_idX
    log_preds = F.log_softmax(logits, dim=-1)
    nll_loss = F.nll_loss(log_preds, labels, ignore_index=PAD_IDX)

    # Mean log softmax ignoring PAD_IDX
    loss = -log_preds.sum(1)
    loss[labels==PAD_IDX] = 0
    loss = loss.mean()

    # Without smoothing, the loss is the same as CrossEntropy ignoring PAD_IDX
    return (1-label_smoothing) * nll_loss + (label_smoothing / vocab_size) * loss


def test_loss_no_PAD() -> None:
    batch_size = 3
    seq_length = 5
    vocab_size = 10

    logits = torch.rand(batch_size, seq_length, vocab_size)
    labels = torch.Tensor([
        [4, 9, 7, 5, EOS_IDX],
        [5, 6, 8, 7, EOS_IDX],
        [6, 7, 4, 6, EOS_IDX],
    ]).long()

    loss_func = TranslationLoss()
    avg_loss = loss_func(logits, labels)
    expected = loss_calc(logits, labels)

    assert np.allclose(avg_loss, expected)


def test_loss_no_PAD_with_label_smoothing() -> None:
    batch_size = 3
    seq_length = 5
    vocab_size = 10
    label_smoothing = 0.1

    logits = torch.rand(batch_size, seq_length, vocab_size)
    labels = torch.Tensor([
        [4, 9, 7, 5, EOS_IDX],
        [5, 6, 8, 7, EOS_IDX],
        [6, 7, 4, 6, EOS_IDX],
    ]).long()

    loss_func = TranslationLoss(label_smoothing)
    avg_loss = loss_func(logits, labels)
    expected = loss_calc(logits, labels, label_smoothing)

    assert np.allclose(avg_loss, expected)


def test_loss_with_PAD() -> None:
    batch_size = 4
    seq_length = 5
    vocab_size = 10

    logits = torch.rand(batch_size, seq_length, vocab_size)
    labels = torch.Tensor([
        [4, 9, 5,       EOS_IDX, PAD_IDX],
        [4, 9, 7,       EOS_IDX, PAD_IDX],
        [5, 6, 8,       7,       EOS_IDX],
        [6, 7, EOS_IDX, PAD_IDX, PAD_IDX],
    ]).long()

    loss_func = TranslationLoss()
    avg_loss = loss_func(logits, labels)
    expected = loss_calc(logits, labels)

    assert np.allclose(avg_loss, expected)


def test_loss_with_PAD_with_label_smoothing() -> None:
    batch_size = 4
    seq_length = 5
    vocab_size = 10
    label_smoothing = 0.1

    logits = torch.rand(batch_size, seq_length, vocab_size)
    labels = torch.Tensor([
        [4, 9, 5,       EOS_IDX, PAD_IDX],
        [4, 9, 7,       EOS_IDX, PAD_IDX],
        [5, 6, 8,       7,       EOS_IDX],
        [6, 7, EOS_IDX, PAD_IDX, PAD_IDX],
    ]).long()

    loss_func = TranslationLoss(label_smoothing)
    avg_loss = loss_func(logits, labels)
    expected = loss_calc(logits, labels, label_smoothing)

    assert np.allclose(avg_loss, expected)