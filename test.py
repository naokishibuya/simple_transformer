import argparse
import os
import torch
import simple_transformer as T


def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(description='Transformer testing')
    parser.add_argument('checkpoint_path', type=str, help='Checkpoint path')
    args = parser.parse_args()

    # Load config and vocab pair
    model_dir = os.path.dirname(args.checkpoint_path)
    config = T.load_config(os.path.join(model_dir, 'config.yaml'))
    src_vocab, tgt_vocab = T.load_vocab_pair(**config.vocab)

    # Build greedy translator using a pretrained transformer
    model = T.make_model(
        input_vocab_size= len(src_vocab),
        output_vocab_size=len(tgt_vocab),
        **config.model)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])

    translator = T.make_translator(
        transformer=model, 
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        **config.translator)

    # Load test dataset and translate
    test_dataset = T.load_dataset(split='test', **config.dataset)
    for src_text, tgt_text in test_dataset:
        print('-'*100)
        output = translator(src_text)
        print(src_text + tgt_text + output)


if __name__=='__main__':
    main()
