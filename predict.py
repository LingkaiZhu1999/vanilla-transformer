import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from torchtext.data.utils import get_tokenizer
from typing import Union
from model import Transformer
from config import config
from dataset import Multi30kDe2En
from torchtext.data.metrics import bleu_score
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

def translate_sentences(sentences: Union[list, str], model: Transformer, src_vocab: Vocab, trg_vocab: Vocab, max_len=50,
                       device='cpu'):
    model.eval()
    if isinstance(sentences[0], str):
        de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
        tokens_batch = [de_tokenizer(sent.lower()) for sent in sentences]
        # tokens_batch = de_tokenizer(sentence.lower())
    else:
        # tokens = [token.lower() for token in sentence]
        tokens_batch = [[token.lower() for token in sent] for sent in sentences]

    tokens_batch = [['<bos>'] + tokens + ['<eos>'] for tokens in tokens_batch]  # add bos and eos tokens to the sides of the sentence
    src_indices = []
    for tokens in tokens_batch:
        indices = [src_vocab[token] for token in tokens]
        src_indices.append(torch.LongTensor(indices))

    # src_tensor = pad_sequence(src_indices, batch_first=True, padding_value=src_vocab['<pad>']).to(device)
    src_tensor = torch.stack(src_indices)
    src_mask = model.src_mask(src_tensor).to(device)

    with torch.no_grad():
        src_encoded = model.encoder(src_tensor, src_mask)
    
    batch_size = src_tensor.size(0)
    trg_indices = torch.full((batch_size, 1), trg_vocab['bos'], dtype=torch.long, device=device) # an empty target sentence to be filled in the following loop
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for i in range(max_len):
        trg_mask = model.trg_mask(trg_indices).to(device)

        with torch.no_grad():
            output = model.decoder(trg_indices, src_encoded, trg_mask, src_mask)
        next_tokens = output.argmax(-1)[:, -1]
        trg_indices = torch.cat([trg_indices, next_tokens.unsqueeze(1)], dim=1)
        
        eos_tokens = (next_tokens == trg_vocab['<eos>'])
        finished |= eos_tokens
        if finished.all():
            break
        output_tokens = []
        for seq in trg_indices:
            tokens = []
            for idx in seq:
                token = trg_vocab.lookup_token(idx.item())
                tokens.append(token)
                if token == '<eos>':
                    break
            output_tokens.append(tokens)

    return output_tokens

def translate_sentence(sentence: Union[list, str], model: Transformer, src_vocab: Vocab, trg_vocab: Vocab, max_len=50,
                       device='cpu'):
    model.eval()
    if isinstance(sentence, str):
        de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
        tokens = de_tokenizer(sentence.lower())
    else:
        tokens = [token.lower() for token in sentence]

    tokens = ['<bos>'] + tokens + ['<eos>']  # add bos and eos tokens to the sides of the sentence
    src_indices = [src_vocab[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    src_mask = model.src_mask(src_tensor).to(device)

    with torch.no_grad():
        src_encoded = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_vocab['<bos>']]  # an empty target sentence to be filled in the following loop

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.trg_mask(trg_tensor).to(device)

        with torch.no_grad():
            output = model.decoder(trg_tensor, src_encoded, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab['<eos>']:
            break

    output_tokens = trg_vocab.lookup_tokens(trg_indexes)

    return output_tokens
     
def fix_punctuation(tokens):
    cleaned = []
    for token in tokens:
        if token in [".", ",", "!"] and cleaned:
            cleaned[-1] += token  # Attach punctuation to previous token
        else:
            cleaned.append(token)
    return cleaned  
    
def test(model, dataloader, trg_vocab, max_length=50, device='cuda'):
    model.eval()
    hypotheses = []
    references = []
    with tqdm(dataloader, unit="batch", desc=f'Evaluating... ',
                bar_format='{desc:<16}{percentage:3.0f}%|{bar:70}{r_bar}', ascii=" #") as iterator:
        with torch.no_grad():
            for src, trg in iterator:
                src, trg = src.to(device), trg.to(device)
                # src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
                src_mask = model.src_mask(src).to(device)
                src_encoded = model.encoder(src, src_mask)
                trg_indexes = [trg_vocab['<bos>']] * dataloader.batch_size # an empty target sentence to be filled in the following loop
                print("trg", trg_indexes[0], trg_indexes[1])
                for i in range(max_length):
                    trg = torch.LongTensor(trg_indexes).to(device)
                    trg_mask = model.trg_mask(trg).to(device)
                    output = model.decoder(trg, src_encoded, trg_mask, src_mask)
                    pred_token = output.argmax(dim=-1)
                    print(pred_token)
                # for i in range(preds.size(0)):
                #     pred_tokens = preds[i].tolist()
                #     target_tokens = trg[i, 1:].tolist()
                #     pred_str = trg_vocab.lookup_tokens(pred_tokens)
                #     target_str = trg_vocab.lookup_tokens(target_tokens)

                #     hypotheses.append(pred_str)
                #     references.append([target_str])
    bleu_scores = bleu_score(hypotheses, references)
    print(bleu_scores)
      
if __name__ == '__main__':
    dataset = Multi30kDe2En('train')
    de_vocab = dataset.de_vocab
    en_vocab = dataset.en_vocab
    config['src_vocab_size'] = len(dataset.de_vocab)
    config['trg_vocab_size'] = len(dataset.en_vocab)
    config['src_pad_idx'] = Multi30kDe2En.PAD_IDX
    config['trg_pad_idx'] = Multi30kDe2En.PAD_IDX
    src_vocab_size = config['src_vocab_size']
    trg_vocab_size = config['trg_vocab_size']
    ff_hid_dim = config['ff_hid_dim']
    embed_dim = config['embed_dim']
    n_blocks = config['n_blocks']
    n_heads = config['n_heads']
    max_length = config['max_length']
    dropout = config['dropout']
    device = config['device']
    src_pad_idx = config['src_pad_idx']
    trg_pad_idx = config['trg_pad_idx']
    lr = config['lr']
    clip = config['clip']
    weights_path = 'weights/50.pt'

    model = Transformer(src_vocab_size,
                        trg_vocab_size,
                        src_pad_idx,
                        trg_pad_idx,
                        embed_dim,
                        n_blocks,
                        n_heads,
                        ff_hid_dim,
                        max_length,
                        dropout,
                        device)
    model.to(device)
    state_dict = torch.load(weights_path, map_location=device)

    # Remove 'module.' prefix from keys
    # state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # # Load the modified state_dict into the model
    # model.load_state_dict(state_dict)
    # model.load_state_dict(torch.load(weights_path, map_location=device))

    sentences = ['Eine Gruppe von Menschen steht vor einem Iglu', 
                'In der Türkei hat die Polizei den Istanbuler Bürgermeister Ekrem İmamoğlu festgenommen.']
    outputs = translate_sentence(sentences[0], model, de_vocab, en_vocab, device=device)
    # print(f'Translation: {" ".join(outputs)}'.replace('<bos>', '').replace('<eos>', ''))
    # print('---------------------')
    # outputs = translate_sentences(sentences, model, de_vocab, en_vocab, device=device)
    # for output in outputs:
    #     print(f'Translation: {" ".join(output)}'.replace('<bos>', '').replace('<eos>', ''))
    
    dataset_test = Multi30kDe2En('test', istoken=False)
    dataloader = DataLoader(dataset_test, batch_size=1)
    # test(model, dataloader, en_vocab, device=device)
    bleu_ = []
    candidates = []
    targets = []
    for src, trg in dataloader:
        # print(src, trg)
        print(src)
        src, trg = src.to(device), trg.to(device)
        output = translate_sentence(src[0], model, de_vocab, en_vocab, device=device)
        # print(f'Translation: {" ".join(output)}'.replace('<bos>', '').replace('<eos>', ''))
        try:
            output.remove('<bos>')
            output.remove('<eos>')
        except:
            None
        candidates.append(fix_punctuation(output))
        targets.append([trg[0].split()])
    print(bleu_score(candidates, targets))
    # print(bleu_ / len(bleu_))
    # avg_bleu = test(model, dataloader, en_vocab, device)
    # print(avg_bleu)
    
 

