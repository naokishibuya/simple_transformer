import torch
from .transformer import Transformer
from ..data import Vocab, SOS_IDX, EOS_IDX


class Translator:
    def __call__(self, text: str) -> str:
        raise Exception('You must implemennt __call__!')


class GreedyTranslator(Translator):
    """ Greedy decoding append the most probable token for the next iteration input to the decoder.
    
    1. Use Encoder to extract features from the input sentence.
    2. The first input to the decoder is SOS (start-of-sentence) marker.
    3. Use Decoder to calculates logits for all target vocab tokens
    4. Choose the most probable token index
    5. If the chosen index is for EOS (end-of-sentence) marker, the translation is done.
    6. Otherwise, concatenate the chosen token to the decoder input and repeat the process.
    """
    def __init__(self, transformer: Transformer, src_vocab: Vocab, tgt_vocab: Vocab, output_length_extra: int) -> None:
        self.transformer = transformer
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.output_length_extra = output_length_extra
    
    def __call__(self, text: str) -> str:
        # Encoder with a batch of one input
        src = torch.Tensor([self.src_vocab(text.strip())])
        features = self.transformer.encode(src)

        # Start with SOS
        tgt = torch.Tensor([[SOS_IDX]]).long()

        max_output_length = src.shape[-1] + self.output_length_extra
        for i in range(max_output_length):
            # Decoder prediction
            logits = self.transformer.decode(features, tgt)

            # Greedy selection
            token_index = torch.argmax(logits[:, -1], keepdim=True)
            if token_index.item()==EOS_IDX: # EOS is most probable => Exit
                break

            # Next Input to Decoder
            tgt = torch.cat([tgt, token_index], dim=1)
        
        # Simple handling of spaces (not the best)
        output = ''
        for i in tgt[0, 1:].numpy():
            token = self.tgt_vocab.tokens[i]
            if len(output)==0 or token in ('.', '!', ',', ';', ':', '\''):
                output += token
            else:
                output += ' ' + token
        return output
