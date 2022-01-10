import argparse
import os
import spacy
from datetime import datetime
from tqdm import tqdm
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import simple_transformer as T


def main() -> None:
    # Command line args
    parser = argparse.ArgumentParser(description='Transformer training')
    parser.add_argument('config_path', type=str, nargs='?', default='config/config.yaml', help='Config YAML path')
    parser.add_argument('--checkpoint_path', type=str, help='Checkpoint path')
    args = parser.parse_args()

    # if checkpoint is provided, override the config path
    if args.checkpoint_path is not None:
        config_dir = os.path.dirname(args.checkpoint_path)
        args.config_path = os.path.join(config_dir, 'config.yaml')

    # Load config and vocab pair
    config = T.load_config(args.config_path)
    src_vocab, tgt_vocab = T.load_vocab_pair(**config.vocab)

    # Build a model
    model = T.make_model(
        input_vocab_size=len(src_vocab),
        output_vocab_size=len(tgt_vocab),
        **config.model)

    if args.checkpoint_path is not None:
        checkpoint_path = os.path.join(args.checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resume training from epoch={start_epoch}')
    else:
        start_epoch = 0

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    # Optimizer, scheduler & loss func
    optimizer = T.make_optimizer(model.parameters(), **config.optimizer)
    scheduler = T.make_scheduler(optimizer, **config.scheduler) if 'scheduler' in config else None
    loss_func = T.make_loss_function(**config.loss).to(device)

    # Tensorboard
    experiment_name = '-'.join([
        datetime.now().strftime('%Y%m%d-%H%M%S'),
        config.dataset.name,
        config.model.name
    ])
    log_dir = os.path.join('runs', experiment_name)
    writer = SummaryWriter(log_dir)

    # keep the copy of config for this training
    config.save(os.path.join(log_dir, 'config.yaml'))

    # Epoch loop
    for epoch in range(start_epoch, config.epochs):
        # train
        train_dataset = T.load_dataset(split='train', **config.dataset)
        train_loader = T.make_dataloader(train_dataset, src_vocab, tgt_vocab, config.batch_size, device)
        train_loss = train(epoch, model, train_loader, loss_func, optimizer, scheduler, writer)
        writer.add_scalar('train/loss', train_loss, epoch)

        # validate
        val_dataset = T.load_dataset(split='valid', **config.dataset)
        val_loader = T.make_dataloader(val_dataset, src_vocab, tgt_vocab, config.batch_size, device)
        val_loss = validate(epoch, model, val_loader, loss_func)
        writer.add_scalar('val/loss', val_loss, epoch)

        # save the model per epoch
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        model_save_path = os.path.join(log_dir, f'checkpoint-{epoch:03d}-{val_loss:0.4f}.pt')
        checkpoint = {'epoch': epoch, 'state_dict': state_dict}
        torch.save(checkpoint, model_save_path)


def train(epoch: int,
          model: nn.Module,
          loader: DataLoader,
          loss_func: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          writer: SummaryWriter) -> float:
    model.train()

    total_loss = 0
    num_batches = len(loader)
    
    with tqdm(loader, unit='batch') as iter:
        iter.set_description(f'Train {epoch}')

        for src_batch, tgt_batch, tgt_label, src_mask, tgt_mask in iter:
            # feed forward
            logits = model(src_batch, tgt_batch, src_mask, tgt_mask)

            # loss calculation
            loss = loss_func(logits, tgt_label)
            total_loss += loss.item()
            iter.set_postfix(loss=loss.item())

            # back-prop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # learning rate scheduler
            if scheduler is not None:
                for i, lr in enumerate(scheduler.get_last_lr()):
                    writer.add_scalar(f'lr-{i}', lr, scheduler._step_count)
                scheduler.step()

    # average training loss
    avg_loss = total_loss / num_batches
    return avg_loss


def validate(epoch: int,
             model: nn.Module,
             loader: DataLoader,
             loss_func: torch.nn.Module) -> float:
    model.eval()

    total_loss = 0
    num_batches = len(loader)

    with tqdm(loader, unit='batch') as iter:
        iter.set_description(f'Valid {epoch}')

        for src_batch, tgt_batch, tgt_label, src_mask, tgt_mask in iter:
            # feed forward
            logits = model(src_batch, tgt_batch, src_mask, tgt_mask)

            # loss calculation
            loss = loss_func(logits, tgt_label)
            total_loss += loss.item()
            iter.set_postfix(loss=loss.item())

    # average validation loss
    avg_loss = total_loss / num_batches
    return avg_loss


if __name__=='__main__':
    main()
