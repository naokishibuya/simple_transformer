import spacy
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from torchtext import datasets
from typing import List, Tuple
from .dataset import load_dataset
from .vocab import Vocab, PAD_IDX, SOS_IDX, EOS_IDX


def make_dataloader(dataset: IterableDataset, 
                    src_vocab: Vocab, 
                    tgt_vocab: Vocab,
                    batch_size: int,
                    device: torch.device) -> DataLoader:
    """ A batch contains a list of text sentence pairs (source sentence, target sentence).

    The collate_fn uses the Vocab objects to tokenize texts and convert them into token indices.

    For target sentences, we prepend SOS_IDX and append EOS_IDX to mark the start of the sentence
    and the end of the sentence, respectively.

    The max sentence length of the batch is used to pad sequences with PAD_IDX.
    """

    def collate_fn(batch: List[Tuple[str, str]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        src_tokens = []
        tgt_tokens = []
        for i, (src_sentence, tgt_sentence) in enumerate(batch):
            src_tokens.append(Tensor(src_vocab(src_sentence)))
            tgt_tokens.append(Tensor([SOS_IDX] + tgt_vocab(tgt_sentence) + [EOS_IDX]))

        # Pad with PAD_IDX up to the max sentence length (within the batch)
        src_batch = pad_sequence(src_tokens, batch_first=True, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_tokens, batch_first=True, padding_value=PAD_IDX)

        tgt_label = tgt_batch[:, 1:]  # Target Labels  =       Target_Sentence_Token_Indices + EOS
        tgt_batch = tgt_batch[:, :-1] # Decoder Inputs = SOS + Target_Sentence_Token_Indices       # aka Outputs (right shifted)

        # Mask indicates where to ignore while calculating attention values
        src_mask, tgt_mask = create_masks(src_batch, tgt_batch)

        # move to the device
        return [x.to(device) for x in [src_batch, tgt_batch, tgt_label, src_mask, tgt_mask]]

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def create_masks(src_batch: Tensor, tgt_batch: Tensor) -> Tuple[Tensor, Tensor]:
    """ Create masks for positions to ignore in attention calculation.

    Args:
        src_batch: source token indicies
        tgt_batch: target token indicies

    Shape:
        - src_batch: (batch_size, max_src_sequence_length) # `max` length is source batch
        - tgt_batch: (batch_size, max_tgt_sequence_length) # `max` length in target batch

    Returns:
        Source mask batch and target mask batch.  
        
        Source mask is for ignoring PAD locations. For example, a source batch with one input [[10, 742, 236, 1, 1, 1]] 
        (PAD_IDX=1) will produce a pad mask [[1, 1, 1, 0, 0, 0]] where 0 means we should ignore the location.

        Target mask is for ignoring PAD locations and the subsequent locations of each token position since
        the decoder should not see the future locations of label sentence.

        For example, a target batch with one input [[510, 49, 13, 1, 1, 1]] will produce a pad mask [[1, 1, 1, 0, 0, 0]].
        Also, for each position, there is a subsequent mask:

        [[1, 0, 0, 0, 0, 0]] # The first token sees only itself for attention calculation.
        [[1, 1, 0, 0, 0, 0]] # The second token sees itself and previous position.
        [[1, 1, 1, 0, 0, 0]] # The third token sees the first three positions.
        [[1, 1, 1, 1, 0, 0]] # And so on.
        [[1, 1, 1, 1, 1, 0]]
        [[1, 1, 1, 1, 1, 1]]

        We combine the pad mask and subsequent mask to create a mask for target sequence.

        PAD mask             &  Subsequent mask     => Target mask
        [[1, 1, 1, 0, 0, 0]] & [[1, 0, 0, 0, 0, 0]] => [[1, 0, 0, 0, 0, 0]]
        [[1, 1, 1, 0, 0, 0]] & [[1, 1, 0, 0, 0, 0]] => [[1, 1, 0, 0, 0, 0]]
        [[1, 1, 1, 0, 0, 0]] & [[1, 1, 1, 0, 0, 0]] => [[1, 1, 1, 0, 0, 0]]
        [[1, 1, 1, 0, 0, 0]] & [[1, 1, 1, 1, 0, 0]] => [[1, 1, 1, 0, 0, 0]]
        [[1, 1, 1, 0, 0, 0]] & [[1, 1, 1, 1, 1, 0]] => [[1, 1, 1, 0, 0, 0]]
        [[1, 1, 1, 0, 0, 0]] & [[1, 1, 1, 1, 1, 1]] => [[1, 1, 1, 0, 0, 0]]

        In the attention calculation, where mask==0 will be given a large negative value and
        softmax will set zero probability to them.

        Also, the target positions with PAD_IDX will be ignored by the loss function which is
        a cross-entropy with ignore_index=PAD_IDX.

        ```
        nn.CrossEntropyLoss(ignore_index=PAD_IDX, ...)
        ```
    """
    # pad mask - set to 1 where we want to process
    src_pad_mask = (src_batch != PAD_IDX).unsqueeze(1) # (batch_size, 1, max_tgt_sequence_length)
    tgt_pad_mask = (tgt_batch != PAD_IDX).unsqueeze(1) # (batch_size, 1, max_src_sequence_length)

    # subsequent mask for decoder inputs
    max_tgt_sequence_length = tgt_batch.shape[1]
    tgt_attention_square = (max_tgt_sequence_length, max_tgt_sequence_length)

    full_mask = torch.full(tgt_attention_square, 1)   # full attention
    subsequent_mask = torch.tril(full_mask)           # subsequent sequence should be invisible to each token position
    subsequent_mask = subsequent_mask.unsqueeze(0)    # add a batch dim (1, max_tgt_sequence_length, max_tgt_sequence_length)

    # The source mask is just the source pad mask.
    # The target mask is the intersection of the target pad mask and the subsequent_mask.
    return src_pad_mask, tgt_pad_mask & subsequent_mask
