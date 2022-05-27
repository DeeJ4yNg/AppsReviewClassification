import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup, DistilBertTokenizer, DistilBertModel
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import re
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings("ignore")

def Data_Sets(data_path, batch_size, Vocab_Path):
    df = pd.read_csv(data_path)  # read data from csv file
    df['Labels'] = df['score'].apply(lambda x: x - 1)
    contents = df.content.values
    # Make score as label
    labels = df.Labels.values

    print('Number of training sentences: {:,}\n'.format(df.shape[0]))

    print('Loading BERT tokenizer...')
    tokenizer = DistilBertTokenizer.from_pretrained(Vocab_Path, do_lower_case=True)

    input_ids = []
    attention_masks = []
    for content in contents:
        encoded_dict = tokenizer.encode_plus(
                        str(content),                      
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))

    train_dataloader = DataLoader(
            train_dataset,
            sampler = RandomSampler(train_dataset),
            batch_size = batch_size
        )

    validation_dataloader = DataLoader(
            val_dataset,
            sampler = SequentialSampler(val_dataset),
            batch_size = batch_size
        )

    return train_dataloader, validation_dataloader, tokenizer

#Data_Sets("./Dataset.csv", 32)


