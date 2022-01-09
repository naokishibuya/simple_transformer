from torch.utils.data import IterableDataset
from torchtext import datasets
from typing import Tuple


def load_dataset(name: str, split: str, language_pair: Tuple[str, str]) -> IterableDataset:
    """ Returns a dataset defined in `torchtext.datasets` namespace.
    (https://github.com/pytorch/text/blob/main/torchtext/datasets/__init__.py)
    
    Args:
        dataset_name: A dataset name. For example, dataset_name could be IWSLT2016, Multi30k, etc.
        split: It can be either train, valid, or test.
        language_pair: A pair of languages. 
                       An example of language_piar is ('de', 'en') for Germany to English translation data.
    """
    dataset_class = eval(f'datasets.{name}')
    dataset = dataset_class(split=split, language_pair=language_pair)
    return dataset
