import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

# softmax attention - based on https://github.com/ischlag/fast-weight-transformers/blob/main/synthetic/softmaxAttention.py
class SoftmaxAttention(nn.Module):
    def __init__(self, embed_dim, d_key, n_keys_values):
        super().__init__()
        # embed_dim is the dimension of the key embedding before projection (d_input)
        # d_key is the dimension of the keys and queries after projection (d_model)
        # n_keys_values is the number of keys/values 
        self.d_key = d_key 
        self.embedding_k = nn.Embedding(num_embeddings=n_keys_values, embedding_dim = embed_dim)

        self.W_k = nn.Linear(embed_dim, d_key)
        self.W_q = nn.Linear(embed_dim, d_key)

        self.init_parameters()
    
    def init_parameters(self):
        nn.init.normal_(self.embedding_k.weight, mean=0., std=1)

        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_q.weight)

    def forward(self, k, q, v_emb):
        # k and q are indices - batch_size x seq_len
        # v_emb is one hot encoded values batch_size x seq_len x v_dim (here, v_dim=seq_len)

        k_emb = self.embedding_k(k) # [batch_size, seq_len, embed_dim]
        q_emb = self.embedding_k(q) # [batch_size, embed_dim]

        K = self.W_k(k_emb) # [batch_size, seq_len, d_key]
        V = v_emb
        Q = self.W_q(q_emb).unsqueeze(1) # [batch_size, 1, d_key]

        A = torch.einsum("bli,bni->bln", K, Q) / np.sqrt(self.d_key) # attention weights
        A = F.softmax(A, dim=1) # normalize over l (number of keys) [batch_size, seq_len, 1] because seq_len keys and 1 query
        v_hat = torch.einsum("bln,bld->bd", A, V) # sum over weighted values and num_querys (which is one) [batch_size, seq_len]

        return v_hat

# linear attention - based on https://github.com/ischlag/fast-weight-transformers/blob/main/synthetic/linearAttention.py
class LinearAttention(nn.Module):
    def __init__(self, embed_dim, d_key, n_keys_values, attention_type='linear', update_rule='sum', arg=0):
        super().__init__()
        # embed_dim is the dimension of the key embedding before projection (d_input)
        # d_key is the dimension of the keys and queries after projection (d_model)
        # n_keys_values is the number of keys/values 
        self.attention_type = attention_type
        self.update_rule = update_rule
        self.arg = arg

        self.d_key = d_key 
        self.embedding_k = nn.Embedding(num_embeddings=n_keys_values, embedding_dim = embed_dim)

        self.W_k = nn.Linear(embed_dim, d_key)
        self.W_q = nn.Linear(embed_dim, d_key)

        self.init_parameters()
    
    def init_parameters(self):
        nn.init.normal_(self.embedding_k.weight, mean=0., std=1)

        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_q.weight)

    def forward(self, k, q, v_emb):
        # k and q are indices - batch_size x seq_len
        # v_emb is one hot encoded values batch_size x seq_len x v_dim (here, v_dim=seq_len)
        
        k_emb = self.embedding_k(k) # [batch_size, seq_len, embed_dim]
        q_emb = self.embedding_k(q) # [batch_size, embed_dim]

        K = self.W_k(k_emb) # [batch_size, seq_len, d_key]
        V = v_emb
        Q = self.W_q(q_emb).unsqueeze(1) # [batch_size, 1, d_key]

        if self.attention_type == 'linear':
            K = F.elu(K) + 1
            Q = F.elu(Q) + 1
        elif self.attention_type == 'dpfp':
            def dpfp(x, nu):
                x = torch.cat([F.relu(x), F.relu(-x)], dim=-1)
                x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, nu+1)], dim=-1) # we take x and then we roll by 1,2&3
                x_repeat = torch.cat([x] * nu, dim=-1) # here we're repeating x enough times to multiply by the rolled versions
                return x_repeat * x_rolled             # then we multiply the rolled versions back by x --- so this is like conjunctive coding!

            K = dpfp(K, self.arg)
            Q = dpfp(Q, self.arg)
        
        if self.update_rule == "sum":
            VK = torch.einsum("blv,blk->bvk", V, K) # sum v k outerproducts to get 'fast weight' matrix
            Z = K.sum(dim=1) # sum keys for norming
            v_hat = torch.einsum("bvp,blp->blv", VK, Q) / (torch.einsum("bp,blp->bl", Z, Q).unsqueeze(-1) + 1e-6) # multiply fast weights by queries batchwise, then norm

        v_hat = v_hat.squeeze(1)

        return v_hat


