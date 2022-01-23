import torch
from torch import Tensor
from .transformer import Transformer
from ..data import Vocab, SOS_IDX, EOS_IDX


class Translator:
    def __init__(self,
                 model: Transformer,
                 source_vocab: Vocab,
                 target_vocab: Vocab,
                 output_length_extra: int) -> None:
        self.model = model
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.output_length_extra = output_length_extra

    def __call__(self, text: str) -> str:
        """ Use Encoder to extract features from the input sentence and decode using translation decoder.
        """
        self.model.eval()
        with torch.no_grad():
            # Encoder with a batch of one input
            enc_inp = torch.Tensor([self.source_vocab(text.strip())])
            enc_out = self.model.encode(enc_inp)

            # maximum output size
            max_output_length = enc_inp.shape[-1] + self.output_length_extra

            return self.decode(enc_out, max_output_length)

    def decode(self, text: str) -> str:
        raise Exception('You must implement decode!')


class GreedyTranslator(Translator):
    """ Greedy decoding append the most probable token for the next iteration input to the decoder.

    2. The first input to the decoder is SOS (start-of-sentence) marker.
    3. Use Decoder to calculates logits for all target vocab tokens
    4. Choose the most probable token index
    5. If the chosen index is for EOS (end-of-sentence) marker, the translation is done.
    6. Otherwise, concatenate the chosen token to the decoder input and repeat the process.
    """
    def __init__(self,
                 model: Transformer,
                 source_vocab: Vocab,
                 target_vocab: Vocab,
                 output_length_extra: int) -> None:
        super().__init__(model, source_vocab, target_vocab, output_length_extra)

    def decode(self, enc_out: Tensor, max_output_length: int) -> str:
        # Start with SOS
        dec_inp = torch.Tensor([[SOS_IDX]]).long()

        for _ in range(max_output_length):
            # Decoder prediction
            logits = self.model.decode(enc_out, dec_inp)

            # Greedy selection
            token_index = torch.argmax(logits[:, -1], keepdim=True)
            if token_index.item()==EOS_IDX: # EOS is most probable => Exit
                break

            # Next Input to Decoder
            dec_inp = torch.cat([dec_inp, token_index], dim=1)

        # text tokens
        dec_out = dec_inp[0, 1:].numpy()
        return [self.target_vocab.tokens[i] for i in dec_out]


class BeamSearchTranslator(Translator):
    """ Beam Search decoding keeps track of the top k (=beam_size) most probable token sequences for each time step.
    """
    def __init__(self,
                 model: Transformer,
                 source_vocab: Vocab,
                 target_vocab: Vocab,
                 output_length_extra: int,
                 beam_size: int, alpha: float) -> None:
        super().__init__(model, source_vocab, target_vocab, output_length_extra)
        self.beam_size = beam_size
        self.alpha = alpha # sequence length penalty

    def decode(self, enc_out: Tensor, max_output_length: int) -> str:
        # Start with SOS
        dec_inp = torch.Tensor([[SOS_IDX]]).long()
        scores = torch.Tensor([0.])
        vocab_size = len(self.target_vocab)
        for i in range(max_output_length):
            # Encoder output expansion from the second time step to the beam size
            if i==1:
                enc_out = enc_out.expand(self.beam_size, *enc_out.shape[1:])

            # Decoder prediction
            logits = self.model.decode(enc_out, dec_inp)
            logits = logits[:, -1] # Last sequence step: [beam_size, sequence_length, vocab_size] => [beam_size, vocab_size]

            # Softmax
            log_probs = torch.log_softmax(logits, dim=1)
            log_probs = log_probs / sequence_length_penalty(i+1, self.alpha)

            # Update score where EOS has not been reched
            log_probs[dec_inp[:, -1]==EOS_IDX, :] = 0
            scores = scores.unsqueeze(1) + log_probs # scores [beam_size, 1], log_probs [beam_size, vocab_size]

            # Flatten scores from [beams, vocab_size] to [beams * vocab_size] to get top k, and reconstruct beam indices and token indices
            scores, indices = torch.topk(scores.reshape(-1), self.beam_size)
            beam_indices  = torch.divide   (indices, vocab_size, rounding_mode='floor') # indices // vocab_size
            token_indices = torch.remainder(indices, vocab_size)                        # indices %  vocab_size

            # Build the next decoder input
            next_dec_inp = []
            for beam_index, token_index in zip(beam_indices, token_indices):
                prev_dec_inp = dec_inp[beam_index]
                if prev_dec_inp[-1]==EOS_IDX:
                    token_index = EOS_IDX # once EOS, always EOS
                token_index = torch.LongTensor([token_index])
                next_dec_inp.append(torch.cat([prev_dec_inp, token_index]))
            dec_inp = torch.vstack(next_dec_inp)

            # If all beams are finished, exit
            if (dec_inp[:, -1]==EOS_IDX).sum() == self.beam_size:
                break

        # convert the top scored sequence to a list of text tokens
        dec_out, _ = max(zip(dec_inp, scores), key=lambda x: x[1])
        dec_out = dec_out[1:].numpy() # remove SOS
        return [self.target_vocab.tokens[i] for i in dec_out if i != EOS_IDX] # remove EOS if exists


def sequence_length_penalty(length: int, alpha: float) -> float:
    """ Sequence length penalty for beam search.
    
    Source: Google's Neural Machine Translation System (https://arxiv.org/abs/1609.08144)
    """
    return ((5 + length) / (5 + 1)) ** alpha
