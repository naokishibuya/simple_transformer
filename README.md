# Simple Transformer

I've written [a series of articles](https://naokishibuya.medium.com/list/language-models-454204ed1217) on the transformer architecture and language models on Medium.

This repository contains an implementation of the Transformer architecture presented in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, et. al.

My goal is to write an implementation that is easy to understand and dig into nitty-gritty details where the devil is.

## Python environment

You can use any Python virtual environment like venv and conda.

For example, with venv:

```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -e.
```

## Spacy Tokenizer Data Preparation

To use Spacy's tokenizer, make sure to download required languages.

For example, English and Germany tokenizers can be downloaded as below:

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Text Data from Torchtext

This project uses text datasets from Torchtext.  

```python
from torchtext import datasets
```

The default configuration uses `Multi30k` dataset.

## Training

```bash
python train.py config_path
```

The default config path is `config/train.small.yaml`.

The train script creates tensorboard log files under `runs`, and saves `config.yaml` and model checkpoints there.

You can run `tensorboard` to see the training progress.

```bash
tensorboard --logdir=runs
```

It is possible to resume training from a checkpoint.

```bash
python train.py --checkpoint_path runs/20220108-164720-Multi30k-Transformer/checkpoint-010-2.3343.pt
```

## Test

```bash
python test.py checkpoint_path
```

Example,

```bash
python test.py runs/20220108-164720-Multi30k-Transformer/checkpoint-010-2.3343.pt
```

The `test.py` resumes with the checkpoint and `config.yaml` in the same directory.

You can specify translator configuration by:

```bash
python test.py checkpoint_path [--config_path config_path]
```

The default is `config/translator.beam.yaml`.

## Unit tests

There are some unit tests in the `tests` folder.

```bash
pytest tests
```

## References:

- [The Annotated Transformer by Harvard NLP](http://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer)
- [How to code The Transformer in Pytorch by Samuel Lynn-Evans](https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec)
- [The Illustrated Transformer by Jay Alammar](http://jalammar.github.io/illustrated-transformer/)
- [Transformer Architecture: The Positional Encoding by Amirhossein Kazemnejad](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- [Transformers without Tears: Improving the Normalization of Self-Attention by Toan Q. Nguyen & Julian Salazar](https://tnq177.github.io/data/transformers_without_tears.pdf)
- [Tensor2Tensor by TensorFlow](https://github.com/tensorflow/tensor2tensor)
- [PyTorch Transformer by PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [Language Modeling with nn.Transformer and Torchtext by PyTorch](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- [My Medium Articles](https://naokishibuya.medium.com/list/language-models-454204ed1217)
