import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from tqdm import tqdm
import numpy as np
import random

from model import Transformer
from utils import AverageMeter
from config import config
from dataset import Multi30kDe2En

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Trainer:
    def __init__(self, config):
        # Configs & Parameters
        self.config = config
        self.src_vocab_size = config['src_vocab_size']
        self.trg_vocab_size = config['trg_vocab_size']
        self.ff_hid_dim = config['ff_hid_dim']
        self.embed_dim = config['embed_dim']
        self.n_blocks = config['n_blocks']
        self.n_heads = config['n_heads']
        self.max_length = config['max_length']
        self.dropout = config['dropout']
        self.device = config['device']
        self.src_pad_idx = config['src_pad_idx']
        self.trg_pad_idx = config['trg_pad_idx']
        self.lr = config['lr']
        self.clip = config['clip']
        self.warmup_steps = config['warmup']
        # Model
        self.model = Transformer(self.src_vocab_size,
                                 self.trg_vocab_size,
                                 self.src_pad_idx,
                                 self.trg_pad_idx,
                                 self.embed_dim,
                                 self.n_blocks,
                                 self.n_heads,
                                 self.ff_hid_dim,
                                 self.max_length,
                                 self.dropout,
                                 self.device)
        self._init_weights()
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        
        if self.warmup_steps!=None:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min((step+1)**-0.5, (step+1) * (self.warmup_steps**-1.5))
            )
        # Loss Function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx, label_smoothing=0.1)
        self.criterion.to(self.device)

        # Metrics
        self.loss_tracker = AverageMeter('loss')

        # Tensorboard
        log_dir = os.path.join(self.config['log_dir'], self.config['name'])
        self.writer = SummaryWriter(log_dir=log_dir)

    def _init_weights(self):
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def train(self, dataloader, epoch, total_epochs):
        self.model.train()
        self.loss_tracker.reset()
        with tqdm(dataloader, unit="batch", desc=f'Epoch: {epoch}/{total_epochs} ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            for src, trg in iterator:
                src, trg = src.to(self.device), trg.to(self.device)
                output = self.model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)

                loss = self.criterion(output, trg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.warmup_steps!=None:
                    self.scheduler.step()

                self.loss_tracker.update(loss.item())
                avg_loss = self.loss_tracker.avg
                iterator.set_postfix(loss=avg_loss)
        return avg_loss

    def evaluate(self, dataloader):
        self.model.eval()
        self.loss_tracker.reset()
        with tqdm(dataloader, unit="batch", desc=f'Evaluating... ',
                  bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
            with torch.no_grad():
                for src, trg in iterator:
                    src, trg = src.to(self.device), trg.to(self.device)
                    output = self.model(src, trg[:, :-1])
                    output_dim = output.shape[-1]
                    output = output.contiguous().view(-1, output_dim)
                    trg = trg[:, 1:].contiguous().view(-1)

                    loss = self.criterion(output, trg)
                    self.loss_tracker.update(loss.item())
                    avg_loss = self.loss_tracker.avg
                    iterator.set_postfix(loss=avg_loss)
        return avg_loss

    def fit(self, train_loader, valid_loader, epochs, trg_vocab):
        for epoch in range(1, epochs + 1):
            print()
            train_loss = self.train(train_loader, epoch, epochs)
            val_loss = self.evaluate(valid_loader)

            # tensorboard
            self.writer.add_scalar('train_loss', train_loss, epoch)
            self.writer.add_scalar('val_loss', val_loss, epoch)
            
            val_src, val_trg = next(iter(valid_loader))
            pred_txt = self.translate_indices(val_src[0].unsqueeze(0).to(self.device), trg_vocab)
            trg_txt = f' Target: {" ".join(trg_vocab.lookup_tokens(val_trg[0].numpy()))}'.replace('<bos>', '').replace('<eos>', '').replace('<pad>', '').replace('<unk>', '')
            
            self.writer.add_text("val_text", pred_txt+trg_txt, epoch)
            should_save_weights = lambda x: not bool(x % self.config['save_interval'])
            if should_save_weights(epoch):
                save_path = os.path.join(self.config['weights_dir'], f'{epoch}.pt')
                torch.save(self.model.state_dict(), save_path)
                print(f'Saved Model at {save_path}')
    
    def translate_indices(self, indices, trg_vocab, max_len=50):
        self.model.eval()
        src_mask = self.model.src_mask(indices).to(self.device)
        with torch.no_grad():
            src_encoded = self.model.encoder(indices, src_mask)
        
        trg_indexes = [trg_vocab['<bos>']]
        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)
            trg_mask = self.model.trg_mask(trg_tensor).to(self.device)
            
            with torch.no_grad():
                output = self.model.decoder(trg_tensor, src_encoded, trg_mask, src_mask)
            
            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)
            
            if pred_token == trg_vocab['<eos>']:
                break
            
        output_tokens = trg_vocab.lookup_tokens(trg_indexes)
        output_tokens = f'Translation: {" ".join(output_tokens)}'.replace('<bos>', '').replace('<eos>', '')
        return output_tokens


if __name__ == '__main__':
    batch_size = config['train_batch_size']

    train_dataset = Multi30kDe2En('train')
    valid_dataset = Multi30kDe2En('valid')

    de_vocab = train_dataset.de_vocab
    en_vocab = train_dataset.en_vocab
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=Multi30kDe2En.collate_fn)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=Multi30kDe2En.collate_fn)

    config['src_vocab_size'] = len(train_dataset.de_vocab)
    config['trg_vocab_size'] = len(train_dataset.en_vocab)
    config['src_pad_idx'] = Multi30kDe2En.PAD_IDX
    config['trg_pad_idx'] = Multi30kDe2En.PAD_IDX
    trainer = Trainer(config)
    trainer.fit(train_loader, valid_loader, config['epochs'], en_vocab)
