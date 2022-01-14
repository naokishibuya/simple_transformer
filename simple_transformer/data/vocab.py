import os
import spacy
from collections import Counter, defaultdict
from torch.utils.data import IterableDataset
from typing import List, Tuple
from .dataset import load_dataset


# special token indices
UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

UNK = '<unk>' # Unknown
PAD = '<pad>' # Padding
SOS = '<sos>' # Start of sentence
EOS = '<eos>' # End of sentence

SPECIAL_TOKENS = [UNK, PAD, SOS, EOS]


class Vocab:    
    """ Dictionary of (token string, token index).
    It can parse an input sentence into a list of token indices. 
    """
    def __init__(self, tokenizer: spacy.language.Language, tokens: List[str]=[]) -> None:
        self.tokenizer = tokenizer
        self.tokens = SPECIAL_TOKENS + tokens
        self.index_lookup = {self.tokens[i]:i for i in range(len(self.tokens))}

    def __len__(self) -> int:
        return len(self.tokens) # vocab size

    def __call__(self, text: str) -> List[int]:
        text = text.strip()
        return [self.to_index(token.text) for token in self.tokenizer(text)]

    def tokenize(self, text: str) -> List[str]:
        text = text.strip()
        return [token.text for token in self.tokenizer(text)]

    def to_index(self, token: str) -> int:
        return self.index_lookup[token] if token in self.index_lookup else UNK_IDX


def load_vocab_pair(
    name: str, 
    language_pair: Tuple[str, str],
    src_language: str,
    tgt_language: str) -> Tuple[Vocab, Vocab]:

    # Load train text pairs
    train_dataset = load_dataset(name, 'train', language_pair)
    src_texts, tgt_texts = list(zip(*train_dataset))

    src_vocab = load_vocab(name, src_language, src_texts)
    tgt_vocab = load_vocab(name, tgt_language, tgt_texts)
    return src_vocab, tgt_vocab


def load_vocab(name: str, language: str, texts: List[str]) -> Vocab:
    tokenizer = spacy.load(language)
    
    path = f'.cache/{name}-{language}-vocab.txt'
    tokens = load_tokens(path)

    if len(tokens)==0:
        print('Generating target vocab...')
        tokens = generate_tokens(tokenizer, texts, path)
        save_tokens(tokens, path)
        print(f'Saved tokens in {path}')

    return Vocab(tokenizer, tokens)


def load_tokens(path: str) -> List[str]:
    tokens = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            tokens = f.read().splitlines()
    return tokens


def generate_tokens(tokenizer: spacy.language.Language, texts: List[str], path: str) -> None:
    # remove new line
    texts = [text.strip() for text in texts]

    # Parse input texts and update the lookup
    counter: Counter = Counter()
    for doc in tokenizer.pipe(texts):
        counter.update([token.text for token in doc])

    # Tokens in frequency order
    return [token for token, count in counter.most_common()]


def save_tokens(tokens: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        f.writelines('\n'.join(tokens))        
    return tokens
