import torch
import torch.nn as nn
from .transformer import Transformer
from .translator import Translator, GreedyTranslator, BeamSearchTranslator


def make_model(name: str, **kwargs) -> nn.Module:
    # Build transformer given a model class name and constructor arguments
    model_class = eval(name)
    model = model_class(**kwargs)
    return model


def make_translator(name: str, **kwargs) -> Translator:
    translator_class = eval(name)
    translator = translator_class(**kwargs)
    return translator
