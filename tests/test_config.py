import os
import sys
script_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(script_dir, '..'))

import simple_transformer as T


def test_config():
    path = os.path.join(script_dir, 'test_config.yaml')
    config = T.load_config(path)

    assert config.model.name          == 'Transformer'
    assert config.model.drop_prob     == 0.1
    assert config.vocab.language_pair == ['de', 'en']
    assert config.optimizer.eps       == 1.0e-9
