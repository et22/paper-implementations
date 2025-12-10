import torch
import torch.nn as nn

def sample_batch(batch_size, seq_length):
    key_ids = torch.stack([torch.randperm(seq_length) for _ in range(batch_size)]) # batch_size x seq_length
    val_onehot = nn.functional.one_hot(key_ids.clone(), num_classes=seq_length).float() # batch_size x seq_length x seq_length

    query_pos = torch.randint(0, seq_length, (batch_size,))    # for each batch element, pick query from selected keys 
    query_ids = key_ids[torch.arange(batch_size), query_pos]

    targets = val_onehot[torch.arange(batch_size), query_pos]

    return key_ids, query_ids, val_onehot, targets

