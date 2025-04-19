#!/usr/bin/env python
# coding: utf-8

# In[32]:


import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k, multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import pandas as pd


# In[ ]:


# (base) C:\Users\Amrita>python -m venv venv

# (base) C:\Users\Amrita>venv\Scripts\activate

# (venv) (base) C:\Users\Amrita>ipython kernel install --user --name=venv
# Installed kernelspec venv in C:\Users\Amrita\AppData\Roaming\jupyter\kernels\venv

# (venv) (base) C:\Users\Amrita>deactivte


# In[50]:


from datasets import load_dataset

ds = load_dataset("bentrevett/multi30k",split="train")


# In[157]:


train_iterator=[]
for i in range(0,100): ## len(ds)/100
    train_iterator.append((ds[i]['de'],ds[i]['en']))

sorted_dataset = sorted(train_iterator, key=lambda x: len(x[0].split()))


# In[159]:


SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Making a placeholder dict to store both tokenizers
token_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Place holder dict for 'en' and 'de' vocab transforms
vocab_transform = {}

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(sorted_dataset, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)


# In[160]:


for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[161]:


def tensor_transform_s(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.flip(torch.tensor(token_ids), dims=(0,)),
                      torch.tensor([EOS_IDX])))

def tensor_transform_t(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func


# In[162]:


text_transform = {}
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_sequences = text_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))
        src_sequences = torch.tensor(src_sequences, dtype=torch.int64)
        tgt_sequences = text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n"))
        tgt_sequences = torch.tensor(tgt_sequences, dtype=torch.int64)
        src_batch.append(src_sequences)
        tgt_batch.append(tgt_sequences)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    src_batch = src_batch.t()
    tgt_batch = tgt_batch.t()
    return src_batch.to(device), tgt_batch.to(device)


# In[163]:


def get_translation_dataloaders(batch_size=4,flip=False):
    ds = load_dataset("bentrevett/multi30k",split="train")
    train_iterator=[]
    for i in range(0,100):
        train_iterator.append((ds[i]['de'],ds[i]['en']))

    sorted_dataset = sorted(train_iterator, key=lambda x: len(x[0].split()))
    sorted_train_iterator = sorted(train_iterator, key=lambda x: len(x[0].split()))
    
    
    valid=load_dataset("bentrevett/multi30k",split="validation")
    valid_iterator=[]
    for i in range(0,len(valid)):
        valid_iterator.append((valid[i]['de'],valid[i]['en']))

    sorted_valid_dataloader = sorted(valid_iterator, key=lambda x: len(x[0].split()))

    
    
    # Update text_transform based on the flip parameter
    if flip:
        text_transform[SRC_LANGUAGE] = sequential_transforms(token_transform[SRC_LANGUAGE], vocab_transform[SRC_LANGUAGE], tensor_transform_s)
    else:
        text_transform[SRC_LANGUAGE] = sequential_transforms(token_transform[SRC_LANGUAGE], vocab_transform[SRC_LANGUAGE], tensor_transform_t)
    text_transform[TGT_LANGUAGE] = sequential_transforms(token_transform[TGT_LANGUAGE], vocab_transform[TGT_LANGUAGE], tensor_transform_t)
    
    train_dataloader = DataLoader(sorted_train_iterator, batch_size=batch_size, collate_fn=collate_fn, drop_last=True)
    valid_dataloader = DataLoader(sorted_valid_dataloader, batch_size=batch_size, collate_fn=collate_fn, drop_last=True)

    return train_dataloader, valid_dataloader


# In[164]:


def index_to_eng(seq_en):
    return " ".join([vocab_transform['en'].get_itos()[index.item()] for index in seq_en])

def index_to_german(seq_de):
    return " ".join([vocab_transform['de'].get_itos()[index.item()] for index in seq_de])


# In[165]:


text_transform = {}
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_sequences = text_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))
        src_sequences = torch.tensor(src_sequences, dtype=torch.int64)
        tgt_sequences = text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n"))
        tgt_sequences = torch.tensor(tgt_sequences, dtype=torch.int64)
        src_batch.append(src_sequences)
        tgt_batch.append(tgt_sequences)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    src_batch = src_batch.t()
    tgt_batch = tgt_batch.t()
    return src_batch.to(device), tgt_batch.to(device)


# In[112]:


# flip=False


# In[113]:


# Update text_transform based on the flip parameter
# if flip:
#     text_transform[SRC_LANGUAGE] = sequential_transforms(token_transform[SRC_LANGUAGE], vocab_transform[SRC_LANGUAGE], tensor_transform_s)
# else:
#     text_transform[SRC_LANGUAGE] = sequential_transforms(token_transform[SRC_LANGUAGE], vocab_transform[SRC_LANGUAGE], tensor_transform_t)
# text_transform[TGT_LANGUAGE] = sequential_transforms(token_transform[TGT_LANGUAGE], vocab_transform[TGT_LANGUAGE], tensor_transform_t)

# # train_dataloader = DataLoader(sorted_train_iterator, batch_size=batch_size, collate_fn=collate_fn, drop_last=True)
# # valid_dataloader = DataLoader(sorted_valid_dataloader, batch_size=batch_size, collate_fn=collate_fn, drop_last=True)


# In[116]:


# src_batch, tgt_batch = [], []
# for src_sample, tgt_sample in batch:
#     src_sequences = text_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))
#     src_sequences = torch.tensor(src_sequences, dtype=torch.int64)
#     tgt_sequences = text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n"))
#     tgt_sequences = torch.tensor(tgt_sequences, dtype=torch.int64)
#     src_batch.append(src_sequences)
#     tgt_batch.append(tgt_sequences)

# src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
# tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
# src_batch = src_batch.t()
# tgt_batch = tgt_batch.t()


# In[145]:


# src=[]


# In[146]:


# sorted_train_iterator[4][0].rstrip("\n")
# text_transform['de'](sorted_train_iterator[4][0].rstrip("\n"))
# token_transform['de'](sorted_train_iterator[4][0].rstrip("\n"))
# vocab_transform['de'](['Eine', 'Berglandschaft'])
# tensor_transform_t([14, 2533]) ## src_sequences
# torch.tensor(tensor_transform_t([14, 2533]))
# src.append(torch.tensor(tensor_transform_t([14, 2533])))


# In[148]:


# sorted_train_iterator[5][0].rstrip("\n")
# text_transform['de'](sorted_train_iterator[5][0].rstrip("\n"))
# token_transform['de'](sorted_train_iterator[5][0].rstrip("\n"))
# vocab_transform['de'](['Eine', 'Mädchengruppe', 'jubelt', '.'])
# tensor_transform_t([   2,  114, 6403, 1694,    4,    3]) ## src_sequences
# torch.tensor(tensor_transform_t([   2,    2,  114, 6403, 1694,    4,    3,    3]))
# src.append(torch.tensor(tensor_transform_t([   2,    2,  114, 6403, 1694,    4,    3,    3])))


# In[149]:


# src


# In[156]:


# pad_sequence(src, padding_value=PAD_IDX, batch_first=True).t()


# In[ ]:




